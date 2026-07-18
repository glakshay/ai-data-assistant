"""Shared fixtures for all test modules."""
import pytest
from unittest.mock import MagicMock


SAMPLE_DEMOGRAPHICS = [
    {"CENSUS_BLOCK_GROUP": "060372001001", "STATE_FIPS": "06", "COUNTY_FIPS": "037",
     "TOTAL_POP": 1500, "MEDIAN_AGE": 34.5, "WHITE_POP": 900, "BLACK_POP": 200,
     "ASIAN_POP": 150, "HISPANIC_POP": 250},
]

SAMPLE_ECONOMICS = [
    {"CENSUS_BLOCK_GROUP": "060372001001", "STATE_FIPS": "06", "COUNTY_FIPS": "037",
     "MEDIAN_INCOME": 75000, "POVERTY_PCT": 12.3, "UNEMPLOYMENT_RATE": 4.5},
]

# Shape returned by schema_cache.discover_schema() / get_tables().
SAMPLE_TABLES = [
    {
        "name": "DEMOGRAPHICS",
        "full_name": '"CENSUS"."PUBLIC"."DEMOGRAPHICS"',
        "columns": ["CENSUS_BLOCK_GROUP", "STATE_FIPS", "COUNTY_FIPS", "TOTAL_POP",
                    "MEDIAN_AGE", "WHITE_POP", "BLACK_POP", "ASIAN_POP", "HISPANIC_POP"],
    },
    {
        "name": "ECONOMICS",
        "full_name": '"CENSUS"."PUBLIC"."ECONOMICS"',
        "columns": ["CENSUS_BLOCK_GROUP", "STATE_FIPS", "COUNTY_FIPS",
                    "MEDIAN_INCOME", "POVERTY_PCT", "UNEMPLOYMENT_RATE"],
    },
]


@pytest.fixture
def mock_snowflake():
    """Mock SnowflakeClient that returns canned rows."""
    client = MagicMock()
    client.execute_query.return_value = SAMPLE_ECONOMICS
    client.get_table_schema.return_value = [
        {"name": "CENSUS_BLOCK_GROUP", "type": "TEXT"},
        {"name": "MEDIAN_INCOME", "type": "NUMBER"},
        {"name": "POVERTY_PCT", "type": "FLOAT"},
    ]
    return client


@pytest.fixture
def sample_tables():
    """Deep-ish copy so a test can mutate columns without leaking to others."""
    return [dict(t, columns=list(t["columns"])) for t in SAMPLE_TABLES]


# Tests mock the provider-agnostic app.llm.complete / app.llm.stream directly
# (see tests/test_agent.py and tests/test_guardrails.py), so no LLM-client fixture is needed.
