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
# Groq first (fast), Gemini next (reliable), then Ollama Cloud (free hosted, one-request-at-a-time),
# NVIDIA/DeepSeek last (accurate but slow and prone to 504s, so it shouldn't block a working backup).
_ALL_PROVIDERS = ["groq", "gemini", "ollama", "nvidia"]

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
    "ollama": {  # Ollama Cloud (hosted, OpenAI-compatible). Free tier = one request at a time.
        # gemma4:31b for BOTH tiers: head-to-head it beat gpt-oss on this pipeline 3/3 vs 0/3, the
        # gpt-oss reasoning models return EMPTY SQL through the OpenAI-compatible endpoint (reasoning
        # eats the token budget). gemma4 is non-reasoning, so it reliably emits SQL and followed the
        # NYC-borough + comparison hints correctly. The strong coders (kimi/deepseek/glm) are paid.
        "main": os.environ.get("OLLAMA_MAIN_MODEL", "gemma4:31b"),
        "fast": os.environ.get("OLLAMA_FAST_MODEL", "gemma4:31b"),
    },
}

_KEY_ENV = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "nvidia": "NVIDIA_API_KEY",
            "ollama": "OLLAMA_API_KEY"}

# Retry only genuinely transient server errors, NOT 429/quota (retrying just burns it).
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


def _fallback_note(failed: str, nxt: str, e: Exception) -> str:
    """A short, user-facing reason for switching providers (for UI visibility)."""
    s = str(e).upper()
    if "429" in s or "RATE LIMIT" in s or "RESOURCE_EXHAUSTED" in s or "QUOTA" in s:
        why = "hit its rate limit"
    elif "504" in s or "TIMEOUT" in s or "DEADLINE" in s or "UNAVAILABLE" in s or "503" in s:
        why = "was unavailable"
    else:
        why = "failed"
    return f"{failed} {why}, switching to {nxt}…"


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
# Per-request timeout so a slow/hung provider can't stall the whole turn, it errors out and the
# fallback moves on. max_retries=0 stops the SDKs from silently retrying (e.g. a 504) before we fall
# through, that internal retrying is what made a dead provider look like a hang.
_TIMEOUT_S = 20.0


def _gemini():
    if "gemini" not in _clients:
        from google import genai
        try:
            from google.genai import types
            _clients["gemini"] = genai.Client(
                api_key=os.environ.get("GEMINI_API_KEY", "MISSING_KEY"),
                http_options=types.HttpOptions(timeout=int(_TIMEOUT_S * 1000)),  # ms
            )
        except Exception:  # older SDK without http_options.timeout, degrade gracefully
            _clients["gemini"] = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "MISSING_KEY"))
    return _clients["gemini"]


def _groq():
    if "groq" not in _clients:
        from groq import Groq
        _clients["groq"] = Groq(api_key=os.environ.get("GROQ_API_KEY", "MISSING_KEY"),
                                timeout=_TIMEOUT_S, max_retries=0)
    return _clients["groq"]


def _nvidia():
    if "nvidia" not in _clients:
        from openai import OpenAI  # NVIDIA NIM is OpenAI-compatible
        _clients["nvidia"] = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ.get("NVIDIA_API_KEY", "MISSING_KEY"),
            timeout=_TIMEOUT_S, max_retries=0,  # fail fast so a 504 falls straight to the next provider
        )
    return _clients["nvidia"]


def _ollama():
    if "ollama" not in _clients:
        from openai import OpenAI  # Ollama Cloud is OpenAI-compatible
        # Longer timeout than the other providers: the free tier loads the model on demand, so the
        # first call after idle is a cold start that can take 30s+ (later calls are fast).
        _clients["ollama"] = OpenAI(
            base_url="https://ollama.com/v1",
            api_key=os.environ.get("OLLAMA_API_KEY", "MISSING_KEY"),
            timeout=60.0, max_retries=0,
        )
    return _clients["ollama"]


def _oai_client(provider):
    if provider == "groq":
        return _groq()
    if provider == "ollama":
        return _ollama()
    return _nvidia()


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
def complete(system, messages, tier="main", temperature=0.0, max_tokens=600, provider=None,
             notices=None) -> str:
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
                logger.warning("LLM '%s' failed (%s), trying next provider", p, str(e)[:80])
                if notices is not None:  # let the caller surface the switch to the UI
                    notices.append(_fallback_note(p, chain[i + 1], e))
                continue
            raise
    raise last


def stream(system, messages, tier="fast", temperature=0.3, max_tokens=700, provider=None,
           notices=None) -> Iterator[str]:
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
                logger.warning("LLM '%s' stream failed (%s), trying next provider", p, str(e)[:80])
                if notices is not None:
                    notices.append(_fallback_note(p, chain[i + 1], e))
                continue
            raise
    if gen is None:
        raise last
    yield from gen
