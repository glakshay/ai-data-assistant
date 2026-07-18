"""
Provider abstraction: the app talks to Groq, Gemini, or NVIDIA (DeepSeek) via one interface.

Neutral interface used by agent.py and guardrails.py:
  complete(system, messages, tier, ...) -> str
  stream(system, messages, tier, ...)   -> Iterator[str]

`messages` is a provider-neutral list of {"role": "user"|"assistant", "content": str};
`system` is the system prompt; `tier` is "main" (SQL/reasoning) or "fast" (routing/chat).

`LLM_PROVIDER` picks the primary; on a rate-limit / quota / auth / transient failure the call
transparently falls through to the next provider that has a key ("best of N"). Model ids per
provider/tier come from env, so a retired id can be swapped without a code change.
"""
import logging
import os
import time
from collections.abc import Iterator

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()

# Fallback order after the primary: the rest of these that have a key configured.
# Groq first (fast), NVIDIA/DeepSeek next (accurate but slow), Gemini last (tight free tier).
_ALL_PROVIDERS = ["groq", "nvidia", "gemini"]

_MODELS = {
    "gemini": {
        "main": os.environ.get("GEMINI_MAIN_MODEL", "gemini-3.5-flash"),
        "fast": os.environ.get("GEMINI_FAST_MODEL", "gemini-3.1-flash-lite"),
    },
    "groq": {
        "main": os.environ.get("GROQ_MAIN_MODEL", "llama-3.3-70b-versatile"),
        "fast": os.environ.get("GROQ_FAST_MODEL", "llama-3.1-8b-instant"),
    },
    "nvidia": {  # build.nvidia.com (OpenAI-compatible). Override ids in .env if needed.
        "main": os.environ.get("NVIDIA_MAIN_MODEL", "deepseek-ai/deepseek-v4-pro"),
        "fast": os.environ.get("NVIDIA_FAST_MODEL", "deepseek-ai/deepseek-v4-flash"),
    },
}

_KEY_ENV = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "nvidia": "NVIDIA_API_KEY"}

# Retry only genuinely transient server errors — NOT 429/quota (retrying just burns it).
_TRANSIENT = ("UNAVAILABLE", "INTERNAL", "503", "500", "DEADLINE", "OVERLOADED")

_clients: dict[str, object] = {}


def active_provider() -> str:
    return DEFAULT_PROVIDER


def model_for(tier: str, provider: str | None = None) -> str:
    provider = (provider or DEFAULT_PROVIDER).lower()
    return _MODELS.get(provider, _MODELS["groq"]).get(tier, _MODELS["groq"]["fast"])


def _has_key(provider: str) -> bool:
    key = os.environ.get(_KEY_ENV.get(provider, ""), "")
    return bool(key) and not key.startswith("your_") and key != "MISSING_KEY"


def _is_transient(e: Exception) -> bool:
    s = str(e).upper()
    return any(t in s for t in _TRANSIENT)


def _fallback_chain(primary: str) -> list[str]:
    ordered = [primary] + [p for p in _ALL_PROVIDERS if p != primary]
    return [p for p in ordered if _has_key(p)]


def _retry(fn):
    for attempt in range(3):
        try:
            return fn()
        except Exception as e:
            if _is_transient(e) and attempt < 2:
                logger.warning("Transient LLM error (retry %d): %s", attempt + 1, str(e)[:120])
                time.sleep(0.6 * (attempt + 1))
                continue
            raise


# ── Clients ───────────────────────────────────────────────────────────────────
def _gemini():
    if "gemini" not in _clients:
        from google import genai
        _clients["gemini"] = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "MISSING_KEY"))
    return _clients["gemini"]


def _groq():
    if "groq" not in _clients:
        from groq import Groq
        _clients["groq"] = Groq(api_key=os.environ.get("GROQ_API_KEY", "MISSING_KEY"))
    return _clients["groq"]


def _nvidia():
    if "nvidia" not in _clients:
        from openai import OpenAI  # NVIDIA NIM is OpenAI-compatible
        _clients["nvidia"] = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ.get("NVIDIA_API_KEY", "MISSING_KEY"),
        )
    return _clients["nvidia"]


def _oai_client(provider):
    return _groq() if provider == "groq" else _nvidia()


def _oai_messages(system, messages):
    out = [{"role": "system", "content": system}]
    for m in messages:
        role = "assistant" if m["role"] == "assistant" else "user"
        out.append({"role": role, "content": m["content"]})
    return out


def _gemini_contents(messages):
    from google.genai import types
    turns = []
    for m in messages:
        role = "model" if m["role"] == "assistant" else "user"
        turns.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    while turns and turns[0].role != "user":  # Gemini requires the first turn to be user
        turns.pop(0)
    return turns


def _gemini_config(system, temperature, max_tokens):
    from google.genai import types
    return types.GenerateContentConfig(
        system_instruction=system, temperature=temperature,
        max_output_tokens=max_tokens, thinking_config=types.ThinkingConfig(thinking_budget=0),
    )


# ── Single-provider calls ─────────────────────────────────────────────────────
def _complete_one(provider, system, messages, tier, temperature, max_tokens) -> str:
    model = model_for(tier, provider)

    def _call():
        if provider == "gemini":
            r = _gemini().models.generate_content(
                model=model, contents=_gemini_contents(messages),
                config=_gemini_config(system, temperature, max_tokens),
            )
            return (r.text or "").strip()
        r = _oai_client(provider).chat.completions.create(
            model=model, messages=_oai_messages(system, messages),
            temperature=temperature, max_tokens=max_tokens,
        )
        return (r.choices[0].message.content or "").strip()

    return _retry(_call)


def _open_stream(provider, system, messages, tier, temperature, max_tokens) -> Iterator[str]:
    """Open the stream EAGERLY (so errors surface here, enabling fallback), return a text iterator."""
    model = model_for(tier, provider)
    if provider == "gemini":
        raw = _retry(lambda: _gemini().models.generate_content_stream(
            model=model, contents=_gemini_contents(messages),
            config=_gemini_config(system, temperature, max_tokens),
        ))

        def gen():
            for chunk in raw:
                yield getattr(chunk, "text", None) or ""
        return gen()

    raw = _retry(lambda: _oai_client(provider).chat.completions.create(
        model=model, messages=_oai_messages(system, messages),
        temperature=temperature, max_tokens=max_tokens, stream=True,
    ))

    def gen():
        for chunk in raw:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            yield delta or ""
    return gen()


# ── Public interface (best-of-N fallback) ─────────────────────────────────────
def complete(system, messages, tier="main", temperature=0.0, max_tokens=600, provider=None) -> str:
    # Explicit provider (e.g. bench.py) → no fallback, so comparisons stay honest.
    if provider is not None:
        return _complete_one(provider.lower(), system, messages, tier, temperature, max_tokens)

    chain = _fallback_chain(DEFAULT_PROVIDER) or [DEFAULT_PROVIDER]
    last = None
    for i, p in enumerate(chain):
        try:
            return _complete_one(p, system, messages, tier, temperature, max_tokens)
        except Exception as e:
            last = e
            if i < len(chain) - 1:  # any error -> try the next provider ("best of N")
                logger.warning("LLM '%s' failed (%s) — trying next provider", p, str(e)[:80])
                continue
            raise
    raise last


def stream(system, messages, tier="fast", temperature=0.3, max_tokens=700, provider=None) -> Iterator[str]:
    if provider is not None:
        yield from _open_stream(provider.lower(), system, messages, tier, temperature, max_tokens)
        return

    chain = _fallback_chain(DEFAULT_PROVIDER) or [DEFAULT_PROVIDER]
    gen = None
    last = None
    for i, p in enumerate(chain):
        try:
            gen = _open_stream(p, system, messages, tier, temperature, max_tokens)
            break
        except Exception as e:
            last = e
            if i < len(chain) - 1:  # any error -> try the next provider
                logger.warning("LLM '%s' stream failed (%s) — trying next provider", p, str(e)[:80])
                continue
            raise
    if gen is None:
        raise last
    yield from gen
