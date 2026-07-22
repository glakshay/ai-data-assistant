"""
Dynamic schema discovery for the SafeGraph / Snowflake "Open Census, Neighborhood
Insights" dataset. Runs at startup and, crucially, loads the dataset's own METADATA:

  - FIELD_DESCRIPTIONS  → maps coded columns (e.g. "B25077e1") to plain English
                          ("Median Value (Dollars)"). This is what makes the cryptic
                          ACS column codes answerable without hardcoding.
  - FIPS_CODES          → state/county names ↔ FIPS codes (for geography joins).

The ACS data tables are named like `2020_CBG_B25` and their columns are CASE-SENSITIVE
quoted identifiers. We keep only the latest year's data tables to avoid 2019/2020 dupes.
"""
import logging
import os
import re
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

# Populated at startup by discover_schema()
_tables: list[dict[str, Any]] = []          # [{name, full_name, columns}]
_field_desc: dict[str, str] = {}            # "B25077e1" -> "Median Value (Dollars)"
_table_topics: dict[str, str] = {}          # full_name -> "Value, Tenure, ..."
_active_year: str | None = None
_fips_table: str | None = None              # full_name of the FIPS codes metadata table
_join_tables: list[dict[str, Any]] = []     # metadata/join tables with their real columns
_table_words: dict[str, set] = {}           # full_name -> word set from its column descriptions


def _q(db: str, schema: str, name: str) -> str:
    return f'"{db}"."{schema}"."{name}"'


def discover_schema(sf) -> None:
    """Discover the latest-year ACS data tables + load field/FIPS metadata. Idempotent."""
    global _tables, _field_desc, _table_topics, _active_year, _fips_table
    db = os.environ.get("SNOWFLAKE_DATABASE", "")
    schema = os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC")

    logger.info("Discovering schema from %s.%s ...", db, schema)
    try:
        rows = sf.execute_query(f'SHOW TABLES IN SCHEMA "{db}"."{schema}"', max_rows=1000)
        all_names = [r.get("name") or r.get("NAME", "") for r in rows if (r.get("name") or r.get("NAME"))]
    except Exception as e:
        logger.error("Schema discovery failed (SHOW TABLES): %s", e)
        _tables = []
        return

    # Pick the most recent year present (e.g. 2020 over 2019).
    years = sorted({n[:4] for n in all_names if n[:4].isdigit()}, reverse=True)
    year = years[0] if years else None
    _active_year = year
    prefix = f"{year}_CBG_" if year else None

    # Keep only that year's ACS data tables: <year>_CBG_B## / _CBG_C## (skip geometry/patterns).
    discovered = []
    for name in all_names:
        if prefix and name.startswith(prefix) and len(name) > len(prefix) and name[len(prefix)] in ("B", "C"):
            full = _q(db, schema, name)
            try:
                # Some ACS tables (e.g. B25 housing) have 1000+ columns, don't truncate,
                # or later subtables like B25077 (median value) get silently dropped.
                col_rows = sf.execute_query(f"DESCRIBE TABLE {full}", max_rows=20000)
                cols = [r.get("name") or r.get("NAME", "") for r in col_rows]
            except Exception:
                cols = []
            discovered.append({"name": name, "full_name": full, "columns": cols})
    _tables = discovered
    logger.info("Discovered %d ACS data tables for year %s.", len(_tables), year)

    _fips_table = _q(db, schema, f"{year}_METADATA_CBG_FIPS_CODES") if year else None
    _load_field_descriptions(sf, db, schema, year)
    _load_join_tables(sf, db, schema, year)
    _build_table_words()


def _build_table_words() -> None:
    """Per-table bag of words from every column description + topic. Used as a deterministic
    routing signal (keyword match) alongside the LLM router, so e.g. 'asian income' reliably
    surfaces the income table (B19) even if the small router model picks the wrong one."""
    global _table_words
    _table_words = {}
    for t in _tables:
        words: set = set()
        for c in t["columns"]:
            d = _field_desc.get(c)
            if d:
                words |= set(re.findall(r"[a-z]{4,}", d.lower()))
        words |= set(re.findall(r"[a-z]{4,}", _table_topics.get(t["full_name"], "").lower()))
        _table_words[t["full_name"]] = words


def _load_join_tables(sf, db: str, schema: str, year: str | None) -> None:
    """Discover the metadata/join tables (names + geography) with their REAL columns, so
    the model can join without inventing column names. Column names are read live, not hardcoded."""
    global _join_tables
    _join_tables = []
    if not year:
        return
    specs = [
        (f"{year}_METADATA_CBG_FIPS_CODES", "State/county names ↔ FIPS codes (for labeling geographies)"),
        (f"{year}_METADATA_CBG_GEOGRAPHIC_DATA", "Per-block-group geography: land/water area, lat/lon (for density)"),
    ]
    for name, note in specs:
        full = _q(db, schema, name)
        try:
            col_rows = sf.execute_query(f"DESCRIBE TABLE {full}", max_rows=2000)
            cols = [r.get("name") or r.get("NAME", "") for r in col_rows]
        except Exception as e:
            logger.warning("Join table %s unavailable: %s", name, e)
            continue
        _join_tables.append({"name": name, "full_name": full, "columns": cols, "note": note})
    logger.info("Loaded %d join/metadata tables.", len(_join_tables))


def _load_field_descriptions(sf, db: str, schema: str, year: str | None) -> None:
    """Build code→label map and per-table topic summaries from FIELD_DESCRIPTIONS."""
    global _field_desc, _table_topics
    _field_desc = {}
    topics: dict[str, set] = defaultdict(set)
    if not year:
        return

    fd_table = _q(db, schema, f"{year}_METADATA_CBG_FIELD_DESCRIPTIONS")
    try:
        rows = sf.execute_query(f"SELECT * FROM {fd_table}", max_rows=100000)
    except Exception as e:
        logger.warning("Field descriptions load failed (grounding degraded): %s", e)
        return

    for r in rows:
        code = r.get("TABLE_ID") or r.get("table_id")
        if not code:
            continue
        title = (r.get("TABLE_TITLE") or "").strip()
        universe = (r.get("TABLE_UNIVERSE") or "").strip()
        # FIELD_LEVEL_1 is usually "Estimate"/"Margin of Error"; deeper levels carry the
        # detail that DISTINGUISHES columns (e.g. "below poverty level"). Lead with those -
        # the shared title is shown once in the table header, so don't let it eat the budget.
        levels = [
            str(r[k]).strip() for k in sorted(r, key=lambda x: x.upper())
            if k.upper().startswith("FIELD_LEVEL_") and r.get(k) not in (None, "")
        ]
        detail = [l for l in levels if l.lower() not in ("estimate", "margin of error")]
        label = " > ".join(detail) if detail else (title or universe)
        _field_desc[code] = label[:160]

        tnum = (r.get("TABLE_NUMBER") or "")[:3]  # e.g. "B25" -> physical table <year>_CBG_B25
        if tnum and title:
            topics[_q(db, schema, f"{year}_CBG_{tnum}")].add(title)

    _table_topics = {k: ", ".join(sorted(v))[:300] for k, v in topics.items()}
    logger.info("Loaded %d field descriptions across %d tables.", len(_field_desc), len(_table_topics))


# ── Accessors (read through these so a startup rebind is always seen) ──────────
def get_tables() -> list[dict[str, Any]]:
    return _tables


def get_table_words() -> dict[str, set]:
    return _table_words


def get_field_desc() -> dict[str, str]:
    return _field_desc


def get_table_topics() -> dict[str, str]:
    return _table_topics


def get_fips_table() -> str | None:
    return _fips_table


def get_join_tables() -> list[dict[str, Any]]:
    return _join_tables


def get_schema_context() -> str:
    """A compact overview: every active table with its human topic summary (for routing)."""
    if not _tables:
        return "Schema not yet loaded."
    lines = [
        f"US Census ACS 5-year estimates ({_active_year}), block-group level. Available tables "
        f"(use the EXACT quoted full name):",
    ]
    for t in _tables:
        topic = _table_topics.get(t["full_name"], "")
        lines.append(f"- {t['full_name']}" + (f", {topic}" if topic else ""))
    return "\n".join(lines)
