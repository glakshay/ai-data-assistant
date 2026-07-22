import re
import logging
from typing import Any

from . import llm

logger = logging.getLogger(__name__)

# SQL/code control characters, treated as a (non-terminal) block.
_INJECTION_PATTERNS = re.compile(
    r"(;\s*(drop|delete|truncate|alter|create|insert|update)\b"
    r"|union\s+select"
    r"|--\s*$"
    r"|\/\*.*\*\/)",
    re.IGNORECASE,
)

# Deterministic backstop for clearly harmful/abusive content. Narrow on purpose (the LLM
# classifier does the nuanced work); this catches the worst even if the model is unavailable.
_INAPPROPRIATE_PATTERNS = re.compile(
    r"\b("
    r"kill\s+(yourself|myself|him|her|them|everyone)"
    r"|how\s+to\s+(kill|murder|poison)\b"
    r"|make\s+a\s+bomb|build\s+a\s+bomb|how\s+to\s+make\s+(a\s+)?(bomb|explosive)"
    r"|child\s*(porn|abuse|sexual)|\bcsam\b"
    r"|rape|molest"
    r")\b",
    re.IGNORECASE,
)

_INAPPROPRIATE_MSG = (
    "I can't help with that, so I've ended this chat. "
    "This assistant only answers questions about US Census data."
)

_SYSTEM = (
    "You are an intent classifier for a US Census data chat assistant. "
    "Reply with EXACTLY one word: CENSUS, CHITCHAT, OFFTOPIC, CLOSING, or INAPPROPRIATE.\n"
    "- INAPPROPRIATE: hate speech, harassment, sexual content, threats or violence, or requests "
    "for clearly illegal or dangerous activity. Choose this even if the message also mentions a "
    "census topic.\n"
    "- CENSUS: the message asks for US population, demographics, income, poverty, employment, "
    "education, housing, geography, migration, or commute data, OR it is a follow-up, correction, "
    "or refinement of a previous census answer (e.g. 'add more detail', 'what about California').\n"
    "- CHITCHAT: greetings or questions about what you are or can do "
    "(e.g. 'hi', 'hello', 'what can you do', 'help'), or gratitude mid-conversation.\n"
    "- CLOSING: the user is wrapping up, a goodbye, 'that's all', 'no thanks, I'm done', or an "
    "affirmative reply confirming they need nothing else (especially right after the assistant "
    "asked whether that was everything).\n"
    "- OFFTOPIC: anything else benign but unrelated, weather, coding help, jokes, other countries' "
    "data, general knowledge."
)


def _history_context(history: list[dict[str, Any]] | None) -> str:
    if not history:
        return ""
    recent = history[-2:]
    return "\n".join(f"{m['role']}: {m['content'][:150]}" for m in recent)


def classify_intent(
    message: str, history: list[dict[str, Any]] | None = None
) -> tuple[str, str]:
    """Classify a user message for routing.

    Returns (intent, guard_message):
      - intent in {"census", "chitchat", "offtopic", "closing", "blocked", "inappropriate"}.
      - guard_message is non-empty for "blocked" (injection / too short) and "inappropriate"
        (the caller shows it and, for "inappropriate", ENDS the chat). Empty otherwise.

    Defense for harmful content: a deterministic regex backstop plus the LLM INAPPROPRIATE
    class. (The earlier Gemini-specific "safety block" signal was dropped when the provider
    became swappable, the regex + classifier are provider-agnostic.)
    """
    text = message.strip()
    if len(text) < 2:
        return "blocked", "Could you type a little more so I can help?"

    if _INAPPROPRIATE_PATTERNS.search(message):
        logger.info("GUARDRAIL ✗ INAPPROPRIATE (regex): %s", message[:80])
        return "inappropriate", _INAPPROPRIATE_MSG

    if _INJECTION_PATTERNS.search(message):
        return "blocked", (
            "Your message looks like it contains SQL/code control characters, so I didn't run it. "
            "Please ask about US Census data in plain language."
        )

    ctx = _history_context(history)
    contents = f"Recent conversation:\n{ctx}\n\nNew message: {message}" if ctx else message

    try:
        verdict = llm.complete(
            _SYSTEM, [{"role": "user", "content": contents}],
            tier="fast", temperature=0, max_tokens=10,
        ).strip().upper()
    except Exception as e:
        logger.warning("Guardrail call failed, defaulting to census: %s", e)
        verdict = "CENSUS"  # fail open so a provider hiccup doesn't block real questions

    if verdict.startswith("INAPPROPRIATE"):
        logger.info("GUARDRAIL ✗ INAPPROPRIATE (model): %s", message[:80])
        return "inappropriate", _INAPPROPRIATE_MSG
    if verdict.startswith("CHITCHAT"):
        intent = "chitchat"
    elif verdict.startswith("CLOSING"):
        intent = "closing"
    elif verdict.startswith("OFFTOPIC"):
        intent = "offtopic"
    else:
        intent = "census"  # default (incl. "CENSUS" / anything unexpected) -> try to answer

    logger.info("GUARDRAIL → %s: %s", intent.upper(), message[:80])
    return intent, ""
