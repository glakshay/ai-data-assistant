import os
import logging
from typing import Any

from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkSessionException

logger = logging.getLogger(__name__)

_QUERY_TIMEOUT_S = 30


class SnowflakeError(Exception):
    pass


class SnowflakeClient:
    def __init__(self):
        self._session: Session | None = None

    def _connect(self) -> Session:
        try:
            return Session.builder.configs({
                "account": os.environ["SNOWFLAKE_ACCOUNT"],
                "user": os.environ["SNOWFLAKE_USER"],
                "password": os.environ["SNOWFLAKE_PASSWORD"],
                "database": os.environ["SNOWFLAKE_DATABASE"],
                "schema": os.environ["SNOWFLAKE_SCHEMA"],
                "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
                "role": os.environ.get("SNOWFLAKE_ROLE", "ACCOUNTADMIN"),
            }).create()
        except Exception as e:
            raise SnowflakeError(f"Failed to connect to Snowflake: {e}") from e

    def session(self) -> Session:
        if self._session is None:
            self._session = self._connect()
        return self._session

    def execute_query(self, sql: str, max_rows: int = 200) -> list[dict[str, Any]]:
        logger.info("Executing SQL: %s", sql[:200])
        try:
            rows = self.session().sql(sql).collect()
            return [r.as_dict() for r in rows[:max_rows]]
        except SnowparkSessionException as e:
            self._session = None  # reset on session errors
            raise SnowflakeError(f"Session error: {e}") from e
        except Exception as e:
            raise SnowflakeError(f"Query failed: {e}") from e

    def get_table_schema(self, table: str) -> list[dict[str, str]]:
        rows = self.execute_query(f"DESCRIBE TABLE {table}")
        return [{"name": r.get("name", ""), "type": r.get("type", "")} for r in rows]

    def close(self):
        if self._session:
            self._session.close()
            self._session = None
