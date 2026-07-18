"""
Unit tests for the text-to-SQL agent pipeline.
Mocks the provider-agnostic LLM layer (app.llm.complete / app.llm.stream) and Snowflake.

Per census turn, llm.complete is called for: routing, SQL generation, follow-ups;
llm.stream is called once for synthesis.
"""
from unittest.mock import patch

CENSUS_DB = '"CENSUS"."PUBLIC"'


def _fresh_meta(**overrides):
    meta = {"offtopic_streak": 0, "answer_count": 0, "winddown_offered": False}
    meta.update(overrides)
    return meta


def _run(history, message, sample_tables, mock_snowflake, *,
         completes=None, stream_chunks=None, intent="census", meta=None):
    if meta is None:
        meta = _fresh_meta()
    completes = list(completes or [])
    stream_chunks = list(stream_chunks or [])
    with patch("app.llm.complete", side_effect=completes) as mc, \
         patch("app.llm.stream", side_effect=lambda *a, **k: iter(list(stream_chunks))) as ms, \
         patch("app.schema_cache._tables", sample_tables):
        from app.agent import run_turn
        chunks = list(run_turn(history, message, mock_snowflake, intent, meta))
    return chunks, mc, ms, meta


# ── census pipeline ───────────────────────────────────────────────────────────

def test_agent_generates_sql_and_answers(mock_snowflake, sample_tables):
    chunks, mc, ms, _ = _run(
        [], "What is the median household income in Texas?", sample_tables, mock_snowflake,
        completes=[
            f'{CENSUS_DB}."ECONOMICS"',
            f'SELECT AVG("MEDIAN_INCOME") FROM {CENSUS_DB}."ECONOMICS" WHERE STATE_FIPS=\'48\'',
            "Compare to California?\nPoverty rate?\nMedian age?",
        ],
        stream_chunks=["The median household income in Texas is about $61,000."],
    )
    mock_snowflake.execute_query.assert_called_once()
    assert "SELECT" in mock_snowflake.execute_query.call_args[0][0].upper()
    combined = " ".join(chunks)
    assert "61,000" in combined or "income" in combined.lower()
    assert any(c.startswith("\n[FOLLOWUPS]") for c in chunks)


def test_agent_asks_for_clarification_on_ambiguous_geo(mock_snowflake, sample_tables):
    chunks, _, _, _ = _run(
        [], "population of Springfield?", sample_tables, mock_snowflake,
        completes=["DEMOGRAPHICS", "CLARIFY: Which state's Springfield do you mean?"],
    )
    assert "springfield" in " ".join(chunks).lower() or "which state" in " ".join(chunks).lower()
    mock_snowflake.execute_query.assert_not_called()


def test_agent_says_cannot_answer_when_unsupported(mock_snowflake, sample_tables):
    chunks, _, _, _ = _run(
        [], "what percent have broadband?", sample_tables, mock_snowflake,
        completes=["DEMOGRAPHICS", "CANNOT_ANSWER: The dataset has no broadband data."],
    )
    assert "broadband" in " ".join(chunks).lower()
    mock_snowflake.execute_query.assert_not_called()


def test_agent_rejects_non_select_output(mock_snowflake, sample_tables):
    chunks, _, _, _ = _run(
        [], "tell me about people", sample_tables, mock_snowflake,
        completes=["DEMOGRAPHICS", "Sure, I think you want demographics data."],
    )
    combined = " ".join(chunks).lower()
    assert "rephrase" in combined or "valid query" in combined
    mock_snowflake.execute_query.assert_not_called()


def test_agent_handles_empty_results(mock_snowflake, sample_tables):
    mock_snowflake.execute_query.return_value = []
    chunks, _, _, _ = _run(
        [], "Population of state 99?", sample_tables, mock_snowflake,
        completes=[
            "DEMOGRAPHICS",
            f'SELECT * FROM {CENSUS_DB}."DEMOGRAPHICS" WHERE STATE_FIPS=\'99\'',
            "q1\nq2\nq3",
        ],
        stream_chunks=["That data is not available at this granularity."],
    )
    mock_snowflake.execute_query.assert_called_once()
    assert "not available" in " ".join(chunks).lower()


def test_agent_handles_snowflake_error_gracefully(mock_snowflake, sample_tables):
    from app.snowflake_client import SnowflakeError
    mock_snowflake.execute_query.side_effect = SnowflakeError("Connection timeout")
    chunks, _, _, _ = _run(
        [], "income in Texas?", sample_tables, mock_snowflake,
        completes=["ECONOMICS", f'SELECT AVG("MEDIAN_INCOME") FROM {CENSUS_DB}."ECONOMICS"'],
    )
    combined = " ".join(chunks).lower()
    assert "error" in combined or "trouble" in combined


def test_agent_graceful_when_schema_missing(mock_snowflake, sample_tables):
    # No tables discovered -> fast-fail before any LLM call.
    chunks, mc, _, _ = _run([], "income in Texas?", [], mock_snowflake)
    assert any("trouble" in c.lower() or "schema" in c.lower() for c in chunks)
    mc.assert_not_called()


def test_agent_handles_llm_error_gracefully(mock_snowflake, sample_tables):
    chunks, _, _, _ = _run(
        [], "income in Texas?", sample_tables, mock_snowflake,
        completes=["ECONOMICS", Exception("Groq API key rejected")],  # non-transient SQL-gen failure
    )
    assert any("trouble" in c.lower() for c in chunks)
    mock_snowflake.execute_query.assert_not_called()


def test_agent_includes_history_in_context(mock_snowflake, sample_tables):
    history = [
        {"role": "user", "content": "What is the income in Texas?"},
        {"role": "assistant", "content": "The median income in Texas is $61,000."},
    ]
    _, mc, _, _ = _run(
        history, "How does that compare to California?", sample_tables, mock_snowflake,
        completes=["ECONOMICS", f'SELECT AVG("MEDIAN_INCOME") FROM {CENSUS_DB}."ECONOMICS"', "q1\nq2\nq3"],
        stream_chunks=["California is higher."],
    )
    # 2nd complete call = SQL generation; its messages (positional arg 1) carry the history.
    sql_messages = mc.call_args_list[1].args[1]
    assert len(sql_messages) >= 3  # 2 history turns + current


# ── conversational: chitchat / off-topic / closing / wind-down ────────────────

def test_chitchat_replies_without_touching_snowflake(mock_snowflake, sample_tables):
    chunks, mc, _, _ = _run(
        [], "hi", sample_tables, mock_snowflake, intent="chitchat",
        stream_chunks=["Hi! I can help you explore US Census data."],
    )
    combined = " ".join(chunks)
    assert "help" in combined.lower() or "hi" in combined.lower()
    mock_snowflake.execute_query.assert_not_called()
    mc.assert_not_called()  # chitchat uses stream only + static chips
    assert any(c.startswith("\n[FOLLOWUPS]") for c in chunks)


def test_chitchat_falls_back_when_llm_returns_nothing(mock_snowflake, sample_tables):
    chunks, _, _, _ = _run([], "hello", sample_tables, mock_snowflake, intent="chitchat",
                           stream_chunks=[])
    combined = " ".join(chunks)
    assert "census" in combined.lower()
    assert any(c.startswith("\n[FOLLOWUPS]") for c in chunks)


def test_interactive_followup_uses_conversation_history(mock_snowflake, sample_tables):
    history = [
        {"role": "user", "content": "Median income in Texas?"},
        {"role": "assistant", "content": "About $61,000."},
    ]
    _, _, ms, _ = _run(
        history, "break it down by county", sample_tables, mock_snowflake,
        completes=["ECONOMICS", f'SELECT "MEDIAN_INCOME", COUNTY_FIPS FROM {CENSUS_DB}."ECONOMICS"', "q1\nq2\nq3"],
        stream_chunks=["Here's a breakdown by county."],
    )
    synth_messages = ms.call_args.args[1]
    assert len(synth_messages) >= 3  # history carried into synthesis


def test_offtopic_soft_deflection_within_limit(mock_snowflake, sample_tables):
    meta = _fresh_meta()
    chunks, mc, _, meta = _run(
        [], "what's the weather?", sample_tables, mock_snowflake, intent="offtopic", meta=meta,
        stream_chunks=["Weather isn't my thing, but I can help with US Census data!"],
    )
    assert "census" in " ".join(chunks).lower()
    mock_snowflake.execute_query.assert_not_called()
    assert meta["offtopic_streak"] == 1


def test_offtopic_escalates_to_firm_after_limit(mock_snowflake, sample_tables):
    meta = _fresh_meta(offtopic_streak=2)
    chunks, _, ms, meta = _run(
        [], "weather again", sample_tables, mock_snowflake, intent="offtopic", meta=meta,
    )
    combined = " ".join(chunks).lower()
    assert "agent bot" in combined or "census data only" in combined
    ms.assert_not_called()  # firm line is canned — no LLM deflection call
    assert meta["offtopic_streak"] == 3


def test_offtopic_streak_resets_on_census(mock_snowflake, sample_tables):
    meta = _fresh_meta(offtopic_streak=2)
    _, _, _, meta = _run(
        [], "income in Texas?", sample_tables, mock_snowflake, meta=meta,
        completes=["ECONOMICS", f'SELECT AVG("MEDIAN_INCOME") FROM {CENSUS_DB}."ECONOMICS"', "q1\nq2\nq3"],
        stream_chunks=["About $61,000."],
    )
    assert meta["offtopic_streak"] == 0


def test_closing_says_goodbye_without_llm_or_sql(mock_snowflake, sample_tables):
    chunks, mc, ms, _ = _run(
        [], "no that's all, thanks", sample_tables, mock_snowflake, intent="closing",
        meta=_fresh_meta(winddown_offered=True),
    )
    assert "rest of your day" in " ".join(chunks).lower()
    mock_snowflake.execute_query.assert_not_called()
    mc.assert_not_called()
    ms.assert_not_called()


def test_winddown_offered_on_third_answer(mock_snowflake, sample_tables):
    meta = _fresh_meta(answer_count=2)
    chunks, _, _, meta = _run(
        [], "income in Texas?", sample_tables, mock_snowflake, meta=meta,
        completes=["ECONOMICS", f'SELECT AVG("MEDIAN_INCOME") FROM {CENSUS_DB}."ECONOMICS"'],
        stream_chunks=["The median income in Texas is about $61,000."],
    )
    combined = " ".join(chunks).lower()
    assert "everything you needed" in combined
    assert meta["answer_count"] == 3
    assert meta["winddown_offered"] is True
    assert not any(c.startswith("\n[FOLLOWUPS]") for c in chunks)  # chips skipped on wrap-up


def test_no_winddown_before_threshold(mock_snowflake, sample_tables):
    meta = _fresh_meta(answer_count=1)
    chunks, _, _, meta = _run(
        [], "income in Texas?", sample_tables, mock_snowflake, meta=meta,
        completes=["ECONOMICS", f'SELECT AVG("MEDIAN_INCOME") FROM {CENSUS_DB}."ECONOMICS"', "q1\nq2\nq3"],
        stream_chunks=["About $61,000."],
    )
    combined = " ".join(chunks).lower()
    assert "everything you needed" not in combined
    assert meta["answer_count"] == 2
    assert meta["winddown_offered"] is False
    assert any(c.startswith("\n[FOLLOWUPS]") for c in chunks)


def test_strip_sql_removes_code_fences():
    from app.agent import _strip_sql
    assert _strip_sql("```sql\nSELECT 1\n```") == "SELECT 1"
    assert _strip_sql("```\nSELECT 2\n```") == "SELECT 2"
    assert _strip_sql("  SELECT 3  ") == "SELECT 3"
