"""
Integration tests against real Snowflake.
Skipped automatically if SNOWFLAKE_ACCOUNT is not set in the environment.
Run with: pytest tests/test_snowflake.py -m integration
"""
import os
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def sf_client():
    if not os.environ.get("SNOWFLAKE_ACCOUNT"):
        pytest.skip("SNOWFLAKE_ACCOUNT not set — skipping integration tests")
    from app.snowflake_client import SnowflakeClient
    client = SnowflakeClient()
    yield client
    client.close()


def test_connection_alive(sf_client):
    rows = sf_client.execute_query("SELECT 1 AS alive")
    assert rows == [{"ALIVE": 1}]


def test_schema_discovery_finds_tables(sf_client):
    """discover_schema() should populate the dynamic table registry from the live DB."""
    from app import schema_cache
    schema_cache.discover_schema(sf_client)
    tables = schema_cache.get_tables()
    assert tables, "No tables discovered — check SNOWFLAKE_DATABASE / SNOWFLAKE_SCHEMA"
    # Each discovered table should carry a fully-qualified name and a column list.
    first = tables[0]
    assert first["full_name"]
    assert isinstance(first["columns"], list)


def test_discovered_table_is_queryable(sf_client):
    """A discovered table should be reachable and return a row count."""
    from app import schema_cache
    schema_cache.discover_schema(sf_client)
    tables = schema_cache.get_tables()
    if not tables:
        pytest.skip("No tables discovered")
    full_name = tables[0]["full_name"]
    rows = sf_client.execute_query(f"SELECT COUNT(*) AS cnt FROM {full_name}")
    assert rows[0]["CNT"] >= 0


def test_empty_result_returns_list(sf_client):
    rows = sf_client.execute_query("SELECT 1 WHERE 1=0")
    assert rows == []


def test_invalid_sql_raises_snowflake_error(sf_client):
    from app.snowflake_client import SnowflakeError
    with pytest.raises(SnowflakeError):
        sf_client.execute_query("SELECT * FROM nonexistent_table_xyz_abc")
