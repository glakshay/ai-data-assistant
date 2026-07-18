"""
Endpoint-level tests for /chat guardrail routing.
Uses FastAPI's TestClient WITHOUT the lifespan context, so no real Snowflake/Gemini
connection is made — the inappropriate/blocked paths return before the agent runs.
"""
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def test_inappropriate_blocks_and_ends_chat():
    import app.main as m
    with patch.object(m, "classify_intent", return_value=("inappropriate", "ENDED-MSG")), \
         patch.object(m, "clear_session") as mock_clear:
        client = TestClient(m.app)
        resp = client.post("/chat", json={"message": "something harmful"})
        body = resp.text

    assert resp.status_code == 200
    assert "ENDED-MSG" in body          # user sees the refusal
    assert "[ENDCHAT]" in body          # UI is told to lock the chat
    mock_clear.assert_called_once()     # server-side session wiped


def test_blocked_injection_does_not_end_chat():
    import app.main as m
    with patch.object(m, "classify_intent", return_value=("blocked", "BLOCK-MSG")), \
         patch.object(m, "clear_session") as mock_clear:
        client = TestClient(m.app)
        resp = client.post("/chat", json={"message": "weird input"})
        body = resp.text

    assert "BLOCK-MSG" in body
    assert "[ENDCHAT]" not in body      # non-terminal — chat continues
    mock_clear.assert_not_called()


def test_offtopic_routes_to_agent_not_terminal():
    import app.main as m
    # Off-topic is handled by the agent (soft/firm), so run_turn should be invoked
    # and the chat should NOT be ended.
    with patch.object(m, "classify_intent", return_value=("offtopic", "")), \
         patch.object(m, "run_turn", return_value=iter(["Weather isn't my thing!"])) as mock_run, \
         patch.object(m, "clear_session") as mock_clear:
        client = TestClient(m.app)
        resp = client.post("/chat", json={"message": "what's the weather?"})
        body = resp.text

    assert "Weather isn't my thing!" in body
    assert "[ENDCHAT]" not in body
    mock_run.assert_called_once()
    mock_clear.assert_not_called()
