# US Census AI Agent

An interactive chat agent that answers natural language questions about the US population using the Snowflake Open Census dataset. The LLM provider is **swappable via `LLM_PROVIDER`** — Groq (Llama 3.3, default) or Google Gemini.

## Demo

**Live URL:** *(update after Railway deploy — copy the public URL here)*

No authentication required — open the URL in any browser and start asking.

## Architecture

```
User → Chat UI (static/index.html)
         │  POST /chat  (SSE stream back)
         ▼
      FastAPI (app/main.py)
         │
    [Intent classifier] ← fast model, context-aware, fast-fail
         │   census / chitchat / offtopic / closing / inappropriate  (sees recent history)
         ├── inappropriate → refuse + END the chat (session wiped)
         ├── offtopic  → friendly deflection ×2, then a firm boundary
         ├── chitchat  → conversational reply + example-question chips
         ├── closing   → "have a great rest of your day"
         ▼
    [Text-to-SQL pipeline] (app/agent.py)
         1. Route to relevant table(s)   ← fast model
         2. Generate SQL (real columns)  ← main model
         3. Execute                      → Snowflake (US Open Census)
         4. Synthesize answer (stream)   ← fast model
         5. Suggest 3 follow-ups         ← fast model
         │
    SSE stream → UI
```

**Provider abstraction + fallback (best of 3).** `app/llm.py` exposes a neutral `complete()` / `stream()` interface with two tiers — **main** (SQL generation) and **fast** (routing, synthesis, classification, chat). `LLM_PROVIDER` selects the primary; on **any provider error (rate-limit, quota, auth, transient) the call transparently falls through to the next provider that has a key**, default order **Groq → NVIDIA (DeepSeek) → Gemini**. So no single provider's free-tier cap breaks the demo. Groq is primary because it's fast (3–6s); DeepSeek `v4-pro` is the most accurate but slow (~30–90s, a reasoning model), so it sits as a fallback rather than the default. Model ids per provider/tier are env-overridable; defaults: Groq `llama-3.3-70b-versatile` / `llama-3.1-8b-instant`, NVIDIA `deepseek-ai/deepseek-v4-pro` / `deepseek-ai/deepseek-v4-flash`, Gemini `gemini-3.5-flash` / `gemini-3.1-flash-lite`. `bench.py` races providers head-to-head on the same grounded pipeline.

**Pattern: Text-to-SQL.** The model generates SQL grounded in the *live, discovered* schema, the backend executes it against Snowflake, and the model synthesizes the rows into a natural-language answer. This is the right pattern for structured tabular data (vs RAG, which is for unstructured documents).

**Schema grounding (comprehensive mapping).** The dataset is SafeGraph Open Census: data tables like `2020_CBG_B25` whose columns are cryptic, case-sensitive ACS codes (`B25077e1`). Naive text-to-SQL fails on it. `app/schema_cache.py` handles this at startup (no hardcoding):
- Discovers all ACS data tables + **every** column (no row cap — early truncation was silently hiding columns like `B25077`).
- Loads the dataset's own **metadata**: `FIELD_DESCRIPTIONS` maps each code → plain English (`B25077e1` → "Median Value (Dollars)"), and the **join tables** `FIPS_CODES` (state/county names) and `GEOGRAPHIC_DATA` (land area → density) are exposed with their real columns so the model never invents column names.
- Routing picks tables by human **topic** (from the metadata), not by code. The SQL step then sees each routed table's columns as `"code" = description`, **relevance-ranked** to the question so the right column is shown even in a 1000-column table — schema visibility without baking columns into the prompt.
- The prompt enforces the two things that make this dataset work: **double-quoting** case-sensitive identifiers, and **FIPS geography** (`CENSUS_BLOCK_GROUP` → state/county via `LEFT`/`SUBSTR` and the FIPS join).

**Interactivity.** The classifier sees recent conversation, so greetings ("hi") get a friendly reply and follow-up corrections ("that's wrong, break it down by county") stay on-topic and reuse context. Per-session counters (in `app/session.py`) drive two behaviors: off-topic messages get up to **2 warm LLM deflections** and then a **firm canned boundary** ("I'm an agent bot for US Census data only") — this cap also bounds LLM spend on spam; and after the **3rd answer** the agent offers a one-time wrap-up ("is that everything for today?"), replying with a goodbye if the user confirms. Ambiguous/underspecified/conflicting questions get a clarifying question; reasonable-but-unanswerable ones get an honest "the dataset doesn't cover that."

## Key decisions

| Decision | Choice | Reason |
|---|---|---|
| Query pattern | Text-to-SQL | Census data is tabular; SQL gives exact grounded answers |
| LLM provider | Groq (default) or Gemini via `LLM_PROVIDER` | Groq is fast + has a generous free tier (won't 429 a demo); Gemini 3.5 is marginally more accurate on hard multi-table joins. Benchmarked head-to-head in `bench.py` |
| Model tiers | main (SQL) + fast (everything else) | Only SQL generation needs the strong model; routing/synthesis/chat run on the cheaper fast tier |
| Schema | Dynamic discovery at startup | No hardcoded tables; adapts to the actual dataset |
| Sessions | In-memory dict | Sufficient for demo; Redis is the obvious prod upgrade |
| Streaming | SSE (not WebSocket) | Server-push only, simpler, HTTP-native |
| Frontend | Vanilla HTML | Zero build tooling, one file, ships fast |
| Deployment | Railway | Push-to-deploy, public URL in minutes |

## Local setup

```bash
git clone <this-repo> && cd snow/
pip install -r requirements.txt   # Python 3.10+ required (uses X | None annotations)
cp .env.example .env
# Fill in .env — at minimum: SNOWFLAKE_* vars + one LLM key:
#   LLM_PROVIDER=groq   (default) -> GROQ_API_KEY   https://console.groq.com/keys
#   LLM_PROVIDER=gemini           -> GEMINI_API_KEY https://aistudio.google.com/apikey
#   LLM_PROVIDER=nvidia           -> NVIDIA_API_KEY https://build.nvidia.com
uvicorn app.main:app --reload
# Open http://localhost:8000
```

`.env` is git-ignored — never commit credentials.

Schema discovery is **automatic** — on startup the app runs `SHOW TABLES` / `DESCRIBE`
against `SNOWFLAKE_DATABASE`.`SNOWFLAKE_SCHEMA` and caches the real table + column
names. There is nothing to hardcode. Just point the `.env` at the mounted dataset.

To sanity-check your dataset manually before first run:

```sql
SHOW TABLES IN SCHEMA "<your_db>"."PUBLIC";
DESCRIBE TABLE "<your_db>"."PUBLIC"."<some_table>";
SELECT COUNT(*) FROM "<your_db>"."PUBLIC"."<some_table>";
```

## Tests

```bash
# Unit tests (no credentials needed)
pytest tests/ -m "not integration" -v

# Integration tests (requires real Snowflake creds in .env)
pytest tests/test_snowflake.py -m integration -v
```

## Provider benchmark (`bench.py`)

Races Groq vs Gemini on the **same grounded pipeline** (identical prompts, schema, and
column-ranking) so the only variable is the model. For each question it reports latency,
the generated SQL, and the answer:

```bash
python bench.py   # needs GROQ_API_KEY + GEMINI_API_KEY in .env
```

In our runs Groq (`llama-3.3-70b`) was faster (3–6s simple, ~30s complex joins) and matched
Gemini on population/value/poverty queries; Gemini 3.5 was slightly more reliable on the
hardest multi-table joins (e.g. population density). Hence **Groq is the default**, with Gemini
a one-env-var switch — or an automatic fallback — away.

**Free-tier caps are real.** Groq is more generous than Gemini's ~5 req/min, but it still has a
**tokens-per-minute** limit, and the grounded SQL prompt is token-heavy — sustained/bursty use
hits it. Mitigations: the SQL prompt is trimmed (relevance-ranked columns, capped descriptions,
other tables listed by name only), and the provider fallback kicks in on a 429. For a demo that
reviewers will hammer, a **paid tier on one provider** removes the cap entirely.

## Deployment (Railway)

1. Push this repo to a private GitHub repo
2. Create a new Railway project → connect the repo
3. Add environment variables (all keys from `.env.example`)
4. Railway auto-detects `railway.toml` and deploys
5. Copy the public URL → update this README

## Edge cases handled

- **Inappropriate content** (hate/harassment/sexual/violence/illegal) → refused **and the chat is ended immediately** (UI locked, session wiped). Defense in depth: a deterministic regex backstop + the LLM `INAPPROPRIATE` class (provider-agnostic)
- **Off-topic questions** → 2 warm deflections, then a firm boundary if the user keeps pushing (escalation counter per session; also caps LLM spend on spam)
- **Greetings / "what can you do"** → friendly conversational reply + example-question chips (no wasted SQL)
- **Wind-down** → after 3 answers the agent offers to wrap up once; a "yes / that's all" gets a goodbye
- **Follow-up corrections** ("that's wrong, break it down by county") → classifier uses history to keep it on-topic; SQL + synthesis reuse the conversation
- **Ambiguous / underspecified / conflicting** (e.g. "Springfield") → the model returns a `CLARIFY:` question instead of guessing
- **Reasonable but unanswerable** → honest `CANNOT_ANSWER:` message, no hallucination
- **Partial matches** → answers the supported part and states what the dataset doesn't cover
- **Empty query results** → agent says the data is unavailable at that granularity
- **Snowflake / schema errors** → descriptive message, not a crash or blank screen
- **SQL injection attempts** → regex check before the LLM classifier ever runs
- **Provider rate-limit / outage** → automatic fallback to the other LLM provider; only if BOTH are capped does the user see an honest "wait a moment" (never a wrong answer)
- **Confident UX** → the UI shows only the answer + figures; SQL, row counts, and the answer are logged to the terminal, not surfaced as confidence-lowering caveats
- **App rate limiting** → 15 requests / minute / session (in-process abuse guard)

## What I'd improve with more time

- Redis-backed sessions and rate limiting (survive restarts, support horizontal scale)
- Cache frequent state-level aggregations (TTL = 24h, data doesn't change)
- **Paid LLM tier** — free tiers have low tokens-per-minute caps that the token-heavy grounded prompt hits under load; the provider fallback + prompt trimming mitigate it, but a paid tier removes it entirely
- **Agentic multi-query loop** — the current single-shot text-to-SQL can't reliably do cross-category math (e.g. population ÷ income across two tables); a tool-use loop that issues and combines several queries would
- Weighted medians instead of the current mean-of-medians approximation for "median" questions
- Progress events during SQL execution ("Querying Snowflake…") so long queries feel responsive
- Eval suite: 50 golden Q&A pairs graded by an LLM judge for regression testing
