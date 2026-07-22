"""
Head-to-head: Gemini vs Groq on the SAME grounded text-to-SQL pipeline.

Reuses the app's real grounding (prompts, schema discovery, field descriptions,
relevance-ranked columns, join tables) so the ONLY variable is the model/provider.
For each question it races both providers concurrently and reports:
  - latency (who answered first)  - the SQL each generated  - the answer (to eyeball accuracy)

Run:  python3.13 bench.py
Needs GEMINI_API_KEY and GROQ_API_KEY in .env (plus the SNOWFLAKE_* vars).
Optional overrides: GEMINI_MAIN_MODEL, GEMINI_FAST_MODEL, GROQ_MAIN_MODEL, GROQ_FAST_MODEL.
"""
import concurrent.futures
import json
import logging
import os
import time

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.ERROR)

from app.snowflake_client import SnowflakeClient
from app import schema_cache
from app.schema_cache import (
    get_field_desc, get_join_tables, get_schema_context, get_table_topics, get_tables,
)
from app.agent import _SQL_SYSTEM, _SYNTH_SYSTEM, _TABLE_SELECT_SYSTEM, _focus_columns, _strip_sql

BUDGET_S = 40
QUESTIONS = [
    "What is the total population of California?",       # truth ~39.3M
    "What is the median home value in Florida?",         # truth ~$260-290k
    "Which 5 counties have the highest poverty rates?",  # needs FIPS name join
    "What is the population density of Texas?",           # needs geographic-area join
    "What percentage of people in New York have a bachelor degree or higher?",
]


# ── Provider adapters (single call interface: system + user -> text) ──────────
def _make_gemini():
    from google import genai
    from google.genai import types
    key = os.environ.get("GEMINI_API_KEY")
    if not key or key.startswith("your_"):
        return None
    client = genai.Client(api_key=key)
    nothink = types.ThinkingConfig(thinking_budget=0)
    main = os.environ.get("GEMINI_MAIN_MODEL", "gemini-3.5-flash")
    fast = os.environ.get("GEMINI_FAST_MODEL", "gemini-3.1-flash-lite")

    def call(system, user, tier, temperature, max_tokens):
        r = client.models.generate_content(
            model=(main if tier == "main" else fast),
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system, temperature=temperature,
                max_output_tokens=max_tokens, thinking_config=nothink,
            ),
        )
        return (r.text or "").strip()

    return {"call": call, "main": main, "fast": fast}


def _make_groq():
    try:
        from groq import Groq
    except ImportError:
        return None
    key = os.environ.get("GROQ_API_KEY")
    if not key or key.startswith("your_"):
        return None
    client = Groq(api_key=key)
    ids = sorted(m.id for m in client.models.list().data)

    def pick(prefs, override):
        if override:
            return override
        for p in prefs:
            for i in ids:
                if p in i.lower():
                    return i
        return ids[0] if ids else None

    main = pick(["llama-3.3-70b", "llama-3.1-70b", "70b", "kimi", "qwen", "deepseek",
                 "llama-4", "gpt-oss"], os.environ.get("GROQ_MAIN_MODEL"))
    fast = pick(["llama-3.1-8b-instant", "8b-instant", "8b", "instant", "gemma"],
                os.environ.get("GROQ_FAST_MODEL"))

    def call(system, user, tier, temperature, max_tokens):
        r = client.chat.completions.create(
            model=(main if tier == "main" else fast),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature, max_tokens=max_tokens,
        )
        return (r.choices[0].message.content or "").strip()

    return {"call": call, "main": main, "fast": fast, "available": ids}


# ── The shared, grounded pipeline (identical for both providers) ───────────────
def run_pipeline(llm, question, sf):
    call = llm["call"]
    tables = get_tables()
    topics = get_table_topics()

    tlist = "\n".join(
        f"- {t['full_name']} | {topics.get(t['full_name']) or ', '.join(t['columns'][:8])}"
        for t in tables
    )
    routed_raw = call(_TABLE_SELECT_SYSTEM, f"Tables:\n{tlist}\n\nQuestion: {question}", "fast", 0.0, 200)
    sel = [l.strip().strip("-").strip() for l in routed_raw.splitlines() if l.strip()]
    routed = [t for t in tables if any(t["name"] in s or t["full_name"] in s for s in sel)][:2] or tables[:2]

    fd = get_field_desc()
    census = bool(fd)
    focus = []
    for t in routed:
        chosen = _focus_columns(t["columns"], question, fd, census)
        lines = [(f'  "{c}" = {fd[c]}' if fd.get(c) else f'  "{c}"') for c in chosen]
        topic = topics.get(t["full_name"], "")
        focus.append(f"TABLE: {t['full_name']}" + (f" (topics: {topic})" if topic else "") + "\n" + "\n".join(lines))
    joins = "\n".join(
        f"{jt['full_name']}, {jt['note']}\n  columns: {', '.join(jt['columns'])}" for jt in get_join_tables()
    )
    sql_system = (
        _SQL_SYSTEM
        + "\n\n=== TABLES ===\n" + get_schema_context()
        + "\n\n=== FOCUS ===\n" + "\n\n".join(focus)
        + "\n\n=== JOIN TABLES ===\n" + joins
    )

    sql = _strip_sql(call(sql_system, f"QUESTION: {question}", "main", 0.0, 600))
    up = sql.upper().lstrip()
    if up.startswith("CLARIFY:") or up.startswith("CANNOT_ANSWER:") or not up.startswith("SELECT"):
        return {"sql": sql, "answer": f"(no query) {sql[:200]}", "rows": 0}

    rows = sf.execute_query(sql, max_rows=200)
    results = json.dumps(rows[:50], default=str)[:6000] if rows else "No rows returned."
    answer = call(_SYNTH_SYSTEM, f"Question: {question}\nSQL: {sql}\nResults: {results}", "fast", 0.3, 700)
    return {"sql": sql, "answer": answer, "rows": len(rows)}


def _timed(llm, question, sf):
    t0 = time.time()
    res = run_pipeline(llm, question, sf)
    return time.time() - t0, res


def race(question, providers, budget=BUDGET_S):
    print("\n" + "=" * 78 + f"\nQ: {question}")
    order = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(providers)) as ex:
        futs = {ex.submit(_timed, p["llm"], question, p["sf"]): name for name, p in providers.items()}
        for fut in concurrent.futures.as_completed(futs, timeout=budget + 10):
            name = futs[fut]
            try:
                dt, res = fut.result()
                order.append((name, dt))
                flag = "⏱️OVER" if dt > budget else "ok"
                print(f"\n  [{name}] {dt:5.1f}s ({flag})  rows={res['rows']}")
                print(f"     SQL: {' '.join(res['sql'].split())[:220]}")
                print(f"     A:   {res['answer'][:320]}")
            except Exception as e:
                print(f"\n  [{name}] FAILED: {str(e)[:180]}")
    if order:
        order.sort(key=lambda x: x[1])
        print(f"\n  >>> FASTEST: {order[0][0]} ({order[0][1]:.1f}s)")


def main():
    gem = _make_gemini()
    groq = _make_groq()
    if not gem and not groq:
        print("No providers configured. Set GEMINI_API_KEY and/or GROQ_API_KEY in .env.")
        return

    # Discover schema once (shared, provider-agnostic), then give each provider its own
    # Snowflake client so concurrent query execution doesn't share one session.
    boot = SnowflakeClient()
    schema_cache.discover_schema(boot)
    print(f"Schema: {len(get_tables())} tables, {len(get_field_desc())} field descriptions.")

    providers = {}
    if gem:
        print(f"Gemini: main={gem['main']}  fast={gem['fast']}")
        providers["gemini"] = {"llm": gem, "sf": SnowflakeClient()}
    else:
        print("Gemini: (skipped, no GEMINI_API_KEY)")
    if groq:
        print(f"Groq:   main={groq['main']}  fast={groq['fast']}")
        providers["groq"] = {"llm": groq, "sf": SnowflakeClient()}
    else:
        print("Groq:   (skipped, no valid GROQ_API_KEY)")

    for q in QUESTIONS:
        race(q, providers)
        time.sleep(12)  # spacing to respect free-tier rate limits


if __name__ == "__main__":
    main()
