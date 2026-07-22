# AI Data Assistant

A conversational AI agent that answers natural-language questions about US public data, population, income, housing, education, and demographics, by turning your question into SQL, running it against a cloud data warehouse, and replying in plain English.

**Live demo:** https://vimsyz.com/ai-data-assistant

## What it does

Ask things like *"What's the median home value in Florida?"* or *"Which counties have the highest poverty rates?"* and get a grounded answer computed from the actual data (not guessed by the model).

## Architecture

```
Browser (chat UI, SSE stream)
        │  POST /chat
        ▼
   FastAPI backend
        │
   [Intent classifier]  → data question / chitchat / off-topic / closing / inappropriate
        │
   [Text-to-SQL pipeline]
        1. Route the question to the relevant table(s)
        2. Generate SQL grounded in the live schema
        3. Execute against the SQL warehouse
        4. Synthesize a natural-language answer from the rows
        5. Suggest follow-ups
        │
   SSE stream → UI
```

**Text-to-SQL, not RAG.** The data is relational, so the right pattern is to generate a SQL query and answer from the real rows, exact, grounded numbers instead of vector-search approximations.

**Schema grounding.** The dataset's columns are opaque codes. At startup the app loads the dataset's own field-description metadata, maps every code to plain English, and relevance-ranks those descriptions per question, so the model is shown the right column instead of guessing one. On top of that, a small **verified metric catalog** pins the exact columns/formulas for the common questions (rent, income, poverty rate, education, internet access, home value, % Hispanic, …), so those are computed correctly rather than guessed, the accuracy lever, at zero extra latency. The same idea grounds **geography**: places that aren't a level in the data are resolved deterministically, cities to their real counties (e.g. NYC to its five boroughs, not just Manhattan or the whole state), and common regions to their state sets (East Coast, Mountain West, …).

**Multi-provider LLM with failover.** A neutral provider layer (`app/llm.py`) runs SQL generation on a stronger model and routing/synthesis/classification on a fast one, across up to four providers (Groq, Gemini, Ollama Cloud, NVIDIA). If a provider is rate-limited, times out, or errors, the call transparently falls through to the next one, with per-request timeouts so a slow provider can't stall the turn, and the switch is surfaced in the UI. One provider's free-tier cap doesn't break the demo.

**Reliability.** Failed SQL self-corrects once (the DB error is fed back for a fix); the warehouse connection self-heals on session expiry; guardrails end the chat on inappropriate input and deflect off-topic questions; answers stream token-by-token over SSE.

## Local setup

```bash
pip install -r requirements.txt   # Python 3.10+
cp .env.example .env              # fill in an LLM key + your warehouse credentials
uvicorn app.main:app --reload
# open http://localhost:8000
```

Set `LLM_PROVIDER` (`groq` | `gemini` | `ollama` | `nvidia`) and the matching API key; the others act as automatic fallbacks when their keys are present.

## Tests

```bash
pytest tests/ -m "not integration" -v   # unit tests, no credentials needed
```

## Known limitations

- **Uncommon aggregations can still be wrong.** Common metrics (verified catalog), common city/region geographies (NYC boroughs, coasts, Mountain West, …), and vs/"and" comparisons are handled deterministically. But questions *outside* those, unusual cross-metric math or a metric that isn't pinned, can still pick the wrong column or mis-aggregate. The synthesis step sanity-checks values (e.g. a percentage must be 0–100) and hedges rather than stating shaky figures. The durable general fix is a **verify-before-answer agentic loop** with a geography/metric resolver and a verification pass; the deterministic catalog + resolvers here are the pragmatic stand-in under free-tier latency constraints.
- **In-memory sessions** reset on restart; Redis is the obvious production upgrade.
- **Free-tier LLM limits** apply; a paid tier removes the per-minute caps that the provider failover works around.

## Honest takeaways

Building this taught me a few things worth writing down:

- **For a data agent, correctness has to be engineered, not assumed.** An LLM will happily write *plausible-but-wrong* SQL over cryptically-coded columns, e.g. picking a "rent-as-%-of-income" count column when asked for rent. Simple lookups worked from day one; the nuanced aggregations are where it slips. Pinning the common metrics to verified columns (the catalog), sanity-checking outputs, and hedging when unsure are the pragmatic fixes; a verify-before-answer loop is the general one.
- **Confident-but-wrong is the worst failure mode.** A number that looks authoritative but is off is more dangerous than an honest "I'm not sure." An early version stripped caveats for a cleaner UX, that was the wrong call for a data tool. Hedging on shaky results is a feature, not a weakness.
- **Grounding beats model size here.** Feeding the model the dataset's own field descriptions (and pinning the important columns) moved the needle far more than swapping to a bigger LLM would have.
