"""
Tests for the intent classifier (guardrail). Mocks the provider-agnostic app.llm.complete.

Contract: classify_intent -> (intent, guard_message)
  intent in {"census", "chitchat", "offtopic", "closing", "blocked", "inappropriate"}
  guard_message is non-empty only for "blocked" and "inappropriate".
"""
import pytest
from unittest.mock import patch


ON_TOPIC = [
    "What is the median household income in California?",
    "Which state has the highest poverty rate?",
    "How many people live in Texas?",
    "What is the unemployment rate in Michigan?",
]
CHITCHAT = ["hi", "hello there", "what can you do?", "help"]
CLOSING = ["no that's all", "that's everything, thanks", "nope, I'm good", "goodbye"]
OFF_TOPIC = [
    "What is the weather in Los Angeles?",
    "Write me a poem about the ocean",
    "Who won the Super Bowl in 2024?",
    "Help me write a Python script for sorting",
]
INJECTION_ATTEMPTS = [
    "SELECT * FROM users; DROP TABLE demographics;--",
    "' OR 1=1; DELETE FROM census --",
    "population UNION SELECT password FROM admin",
]
INAPPROPRIATE_REGEX = ["kill yourself", "how to make a bomb", "how to kill someone"]


def _patch(verdict=None, error=None):
    if error is not None:
        return patch("app.llm.complete", side_effect=error)
    return patch("app.llm.complete", return_value=verdict)


@pytest.mark.parametrize("q", ON_TOPIC)
def test_on_topic_routes_to_census(q):
    with _patch("CENSUS"):
        from app.guardrails import classify_intent
        intent, hard = classify_intent(q)
        assert intent == "census" and hard == ""


@pytest.mark.parametrize("q", CHITCHAT)
def test_greetings_route_to_chitchat(q):
    with _patch("CHITCHAT"):
        from app.guardrails import classify_intent
        intent, _ = classify_intent(q)
        assert intent == "chitchat"


@pytest.mark.parametrize("q", CLOSING)
def test_wrapups_route_to_closing(q):
    with _patch("CLOSING"):
        from app.guardrails import classify_intent
        intent, _ = classify_intent(q)
        assert intent == "closing"


@pytest.mark.parametrize("q", OFF_TOPIC)
def test_off_topic_routes_to_offtopic(q):
    with _patch("OFFTOPIC"):
        from app.guardrails import classify_intent
        intent, hard = classify_intent(q)
        assert intent == "offtopic" and hard == ""


@pytest.mark.parametrize("q", INJECTION_ATTEMPTS)
def test_injection_patterns_blocked_before_llm(q):
    with patch("app.llm.complete", return_value="CENSUS") as mc:
        from app.guardrails import classify_intent
        intent, hard = classify_intent(q)
        assert intent == "blocked" and hard != ""
        mc.assert_not_called()


@pytest.mark.parametrize("q", INAPPROPRIATE_REGEX)
def test_inappropriate_regex_blocks_before_llm(q):
    with patch("app.llm.complete", return_value="CENSUS") as mc:
        from app.guardrails import classify_intent
        intent, hard = classify_intent(q)
        assert intent == "inappropriate" and hard != ""
        mc.assert_not_called()


def test_inappropriate_via_model_verdict():
    with _patch("INAPPROPRIATE"):
        from app.guardrails import classify_intent
        intent, hard = classify_intent("a nasty message the model flags")
        assert intent == "inappropriate" and hard != ""


def test_empty_message_blocked():
    with _patch("CENSUS"):
        from app.guardrails import classify_intent
        intent, hard = classify_intent("  ")
        assert intent == "blocked" and hard != ""


def test_fail_open_on_api_error_defaults_to_census():
    with _patch(error=Exception("provider down")):
        from app.guardrails import classify_intent
        intent, _ = classify_intent("What is the population of Ohio?")
        assert intent == "census"


def test_history_context_passed_to_classifier():
    history = [
        {"role": "user", "content": "Median income in Texas?"},
        {"role": "assistant", "content": "About $61,000."},
    ]
    with patch("app.llm.complete", return_value="CENSUS") as mc:
        from app.guardrails import classify_intent
        classify_intent("add more detail", history)
    # messages positional arg (index 1) is a 1-item list whose content holds the context.
    sent = mc.call_args.args[1][0]["content"]
    assert "Median income in Texas" in sent
    assert "add more detail" in sent
