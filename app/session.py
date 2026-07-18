from collections import defaultdict
from typing import Any

# session_id -> list of {"role": "user"|"assistant", "content": str}
_store: dict[str, list[dict[str, Any]]] = defaultdict(list)

_MAX_HISTORY_TURNS = 20  # keep last 20 turns per session to bound token cost


def _new_meta() -> dict[str, Any]:
    """Per-session conversational counters that drive the interactive behavior."""
    return {
        "offtopic_streak": 0,    # consecutive off-topic messages (escalation)
        "answer_count": 0,       # census answers delivered (wind-down trigger)
        "winddown_offered": False,  # have we offered to wrap up yet?
    }


# session_id -> meta counters. Same in-memory lifetime as _store; Redis in prod.
_meta: dict[str, dict[str, Any]] = defaultdict(_new_meta)


def get_history(session_id: str) -> list[dict[str, Any]]:
    return list(_store[session_id])


def get_meta(session_id: str) -> dict[str, Any]:
    """Return the mutable meta dict for this session (created on first access)."""
    return _meta[session_id]


def append_turn(session_id: str, user_msg: str, assistant_msg: str) -> None:
    history = _store[session_id]
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})
    # Trim to last N turns (each turn = 2 messages)
    if len(history) > _MAX_HISTORY_TURNS * 2:
        _store[session_id] = history[-(_MAX_HISTORY_TURNS * 2):]


def clear_session(session_id: str) -> None:
    _store.pop(session_id, None)
    _meta.pop(session_id, None)
