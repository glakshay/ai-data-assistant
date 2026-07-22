# Reflection

## Development process and key architectural decisions

I read the brief top-to-bottom before writing anything. The dataset is tabular, ACS block-group estimates with hundreds of columns per table and coded identifiers like `B25077e1`. That one fact shaped every decision that followed.

**Why text-to-SQL and not RAG.** RAG is the right call when your source is a document corpus and the answer might live across multiple embedded chunks. This isn't that. The data is relational and the answer is a SQL aggregate; embedding census rows doesn't tell you that `B25077e1` means "Median Value (Dollars)." You'd get similarity scores back, not a ground-truth count or median. Text-to-SQL generates a query from the live schema, runs it against Snowflake, and synthesizes from the actual returned rows. The answer is always grounded in the data, not an approximation from a vector search.

**Why no LangChain.** LangChain's SQL agent would've added indirection without adding value here. The pipeline has five discrete steps: route to the right table, generate SQL, execute against Snowflake, synthesize the rows into a natural answer, suggest follow-ups. Each step needs specific temperature and model-tier settings. LangChain hides exactly the things you need to tune, adds latency per chain step, and makes debugging painful when ACS column case-sensitivity errors surface. Direct API calls keep the control surface visible. I can see the exact prompt, the exact response, and fix it fast.

**The ACS grounding problem.** This is where naive text-to-SQL breaks on this dataset. The column codes are completely opaque, so the model hallucinates names it has never seen. `schema_cache.py` handles this at startup with no hardcoding: it loads the dataset's own `FIELD_DESCRIPTIONS` metadata table, maps each code to plain English, then relevance-ranks those descriptions against the user's question so the right column surfaces first even inside a 1,000-column table. The SQL model then sees `"B25077e1" = Median Value (Dollars)` instead of a meaningless code. That mapping is what makes the whole thing work on real census questions.

**Three-provider fallback.** Single-provider reliance was too risky given the 60-second SLA. Groq runs first: fast (3-6s), generous free tier. NVIDIA/DeepSeek sits second; it's the most accurate on hard multi-table joins but can take 30-90 seconds as a reasoning model, so it wasn't a candidate for primary. Gemini is third, tighter rate limits at around 5 req/min. `llm.py` wraps all three behind a unified `complete()` / `stream()` interface, and any provider error (rate-limit, auth failure, transient 503) falls through to the next one with a key configured. If all three are capped simultaneously, the user gets an honest "wait ~30 seconds" rather than a silent timeout or, worse, a wrong answer from a degraded model.

**Two model tiers.** SQL generation needs the strong model; routing a question to the right table, synthesizing rows into a readable answer, classifying intent, and generating follow-ups do not. Running everything on the main model would burn free-tier token quota about 4x faster than necessary. The `tier="fast"` path uses `llama-3.1-8b-instant` on Groq (sub-second for classification and chitchat) while `llama-3.3-70b-versatile` handles SQL, where a wrong column name crashes the query.

**Temperature settings reflect the task.** SQL generation and table routing sit at `temperature=0`: deterministic, no hallucination. Synthesis runs at `0.3`, enough fluency to not read like a raw query result without drifting from the data. Chitchat and off-topic deflection use `0.5`, which reads noticeably warmer. Follow-up question suggestions get `0.7` so the same three chips don't appear every turn.

**State management.** Each session stores two things in memory: `history` (last 20 turns, trimmed to keep token cost bounded) and `meta` counters (`offtopic_streak`, `answer_count`, `winddown_offered`). The guardrail classifier gets the last 2 turns of history so a message like "add more detail" reads as a census correction, not an off-topic request. The meta counters drive the interactive behaviors: up to 2 warm LLM deflections on off-topic messages, then a firm canned boundary that also caps LLM spend on spam. After the third answer the agent offers a one-time wrap-up. Both counters mutate in-place inside `run_turn`, so `main.py` doesn't need to track them separately.

**Conversation persona.** The system prompts push toward warm over clinical. The synthesis prompt explicitly strips technical scaffolding: no mention of ACS codes, FIPS identifiers, block groups, or data year, so the answer reads like a person giving you a figure rather than a database dumping results. Off-topic deflection is friendly the first couple of turns, then firm but not cold after that. The closing message is a genuine "have a great rest of your day," not "session terminated."

**No auth on the frontend.** The brief said auth is fine if credentials are included. I skipped it deliberately; reviewers shouldn't need to log in to evaluate a demo. Session continuity uses an `httponly`, `samesite=lax` UUID cookie, which is enough for rate-limiting and conversation history without a login wall.

**Streaming via SSE.** Server-sent events fit cleanly: server pushes, client reads, HTTP-native, no WebSocket handshake overhead. Synthesis streams tokens as they arrive so a 20-second Snowflake query still feels live. The `[FOLLOWUPS]` and `[ENDCHAT]` sentinels ride the same SSE stream as typed events, no secondary channel needed.

---

## What I'd improve with more time

**Better SQL-to-text.** The synthesis prompt suppresses all caveats to keep the UX confident, but it occasionally oversimplifies. When the answer is a mean-of-medians (aggregating AVG across block groups), the user can't tell. I'd tune toward "confident but complete": a soft "approximately" for known approximations, without reverting to database-report language.

**Better prompt encoding for the SQL step.** Column descriptions truncate at 70 characters and arrive as flat `"code" = description` pairs. A richer format, nested by topic with sibling columns grouped under their table title, would help the model treat families of related columns (all the B19001 income bracket bins, for example) as a coherent group rather than independent items. That matters most on multi-column aggregations.

**Agentic multi-query loop.** Single-shot SQL can't reliably do cross-category math. Population density is population from one table divided by land area from another, and the current approach sometimes gets that wrong. A tool-use loop that issues and recombines several queries would handle it properly. I deliberately left this out: under a 60-second SLA, an unbounded loop is a reliability risk, and single-shot is more predictable under time pressure. With more time I'd cap the loop at 3 iterations with an explicit bail-out path.

**Redis-backed sessions.** In-memory sessions reset on every Railway restart. Redis with a 24-hour TTL would survive restarts, scale horizontally, and let the rate limiter work correctly across instances. Right now `_rate_store` is per-process, so a load-balanced deploy could exceed the per-session limit.

**Progress events during SQL execution.** Long Snowflake queries (county-level multi-join, 30+ seconds) sit in silence from the user's perspective. A "Querying Snowflake..." SSE event sent right after the SQL step kicks off would make the wait feel shorter.

**A paid LLM tier.** Free tiers have low tokens-per-minute caps and the grounded SQL prompt is token-heavy. The provider fallback and prompt trimming handle it for now, but a single paid tier on one provider removes the cap entirely.

---

## Failure mode analysis

| Failure | How it happens | Mitigation in place | What's missing |
|---|---|---|---|
| **SQL hallucination** | Model invents a column name not in the schema | Columns listed explicitly; prompt says "ONLY the columns listed below"; on a failed query the Snowflake error + bad SQL are fed back for one self-correction retry | A *pre-execution* EXPLAIN/VALIDATE dry-run (the retry only fires after Snowflake rejects the query), plus a unit test for the self-heal path |
| **LLM hang (not an error)** | Provider accepts the request but never responds | 3-retry loop on transient 500/503 errors | `asyncio.wait_for()` around `run_in_executor`; currently no ceiling on how long a provider can hang |
| **Rate-limit cascade** | All three providers 429 simultaneously under heavy load | Per-session rate limit (15 req/min) plus provider fallback chain | Exponential backoff with retry-after header parsing; a circuit breaker per provider |
| **Overly permissive Snowflake role** | `ACCOUNTADMIN` has write access; a prompt injection past the regex could craft a destructive query | SQL injection regex + LLM generates SELECT-only by instruction | Snowflake user should be read-only, scoped to the census schema; ACCOUNTADMIN is a local dev placeholder |
| **Session collision across tabs** | Two tabs with the same session cookie see the same conversation state | httponly + samesite=lax cookie | No per-tab isolation; concurrent sends can interleave confusingly |
| **Mean-of-medians approximation** | "Median income in California" aggregates AVG across block groups: directionally right, not mathematically exact | Synthesis prompt doesn't claim precision | A Snowflake-side PERCENTILE_DISC UDF or an explicit "~" marker in answers where AVG is used |
| **Stale schema after dataset update** | Dataset version bumps; `discover_schema()` only runs at startup | Dynamic discovery, no hardcoded names | A TTL-based re-discovery or a `/reload-schema` admin endpoint |

---

## Testing approach and what I'd add

The decision was to mock at `app.llm.complete` / `app.llm.stream` rather than at the HTTP client. Mocking at the HTTP layer would mean patching different internals per provider (Groq SDK, Gemini SDK, OpenAI SDK), which is brittle and provider-specific. Mocking at the provider-agnostic interface means the agent's actual logic gets exercised: routing decisions, SQL validation, meta counter updates, history propagation, streaming format. The full suite runs in about 2 seconds with no live connections.

`test_guardrails.py` covers the intent classifier contract across on-topic, chitchat, closing, off-topic, injection-blocked, regex-inappropriate, model-flagged inappropriate, empty message, and fail-open on API error. The injection and inappropriate tests assert `mc.assert_not_called()`, verifying the defense-in-depth ordering rather than just the output label.

`test_agent.py` covers the full pipeline: happy path SQL and synthesis, CLARIFY path, CANNOT_ANSWER path, malformed SQL (non-SELECT output), empty Snowflake results, SnowflakeError propagation, LLM error propagation, history carried into SQL and synthesis calls, off-topic streak escalation and reset, closing with no LLM or SQL calls made, winddown trigger at answer count 3, and winddown suppression below the threshold.

`test_main.py` is endpoint-level: inappropriate input ends the chat and calls `clear_session`; a blocked injection doesn't; off-topic routes to `run_turn` without triggering end-chat.

`test_snowflake.py` is integration-only (`@pytest.mark.integration`, skipped without real creds): connection alive, schema discovery finds tables, a discovered table is queryable, empty results return cleanly, and invalid SQL raises `SnowflakeError`.

What I'd add: a golden eval suite with 50 Q&A pairs scored by an LLM judge (accuracy 0/1, hallucination 0/1) run nightly on the deployed app. Unit tests mock the LLM, so they can't catch when a new provider version starts generating wrong SQL on real data. End-to-end Playwright tests against the deployed URL would cover the SSE stream format and assert actual answer content. Load tests with Locust at 30 concurrent sessions would confirm the rate limiter and provider fallback hold under sustained traffic. And a unit test for the SQL self-heal path: mock the LLM to return a query with a non-existent column, then a valid one, and assert the agent feeds the error back and recovers. The self-heal (one corrected retry after a Snowflake error) is in place but currently untested, and it corrects reactively, a pre-execution EXPLAIN/validate would catch the bad column without a Snowflake round-trip.

---

## Product tradeoffs and time allocation

Schema grounding got the most time: the ACS code-to-description mapping, the relevance-ranked column selection, and getting FIPS geography joins right. That's where naive text-to-SQL breaks completely, and getting it solid makes everything else feel reliable. Next was the guardrail classifier and conversational state, since those separate a data demo from something you'd actually hand to a customer. The UI came last and stayed simple: vanilla HTML, no build step, ship it.

Things I deliberately left out: streaming progress events, a proper eval harness, Redis, weighted medians, and a login screen. None of them block a reviewer from evaluating the core functionality.

---

## Security

The Snowflake connection uses `ACCOUNTADMIN` in `.env.example`, a placeholder for local dev only. In production this would be a read-only service account scoped to the census schema, which makes SQL injection structurally a non-issue even if both the regex check and LLM classifier somehow fail.

CORS is `allow_origins=["*"]` for the demo; a production deploy would lock that to the deployed domain. The SSE stream escapes newlines as `\n` literals before sending, and the frontend re-expands them client-side. The HTML renderer uses `textContent` throughout rather than `innerHTML`, so XSS via a crafted census answer isn't possible.

The rate limiter (15 req/min per session) is in-process. It's an abuse guard, not a hard security control; a single actor with many session IDs bypasses it. Redis with a shared global counter would fix that.

---

## Evals and observability

Right now everything goes to structured stdout: intent, SQL, row count, and answer logged at INFO with a session ID prefix, captured by Railway. That's enough to manually debug a bad response.

For production I'd want JSON-structured logs per turn (query_id, intent, provider_used, latency_ms, row_count, answer_len) feeding a dashboard tracking p50/p95 latency, provider fallback rate (a rising rate signals a primary provider degrading before users start complaining), and the off-topic/inappropriate rejection rate. The nightly eval would run `bench.py`-style against 50 golden pairs scored by an LLM judge, catching model regressions before anyone notices them in the chat.
