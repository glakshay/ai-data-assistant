import os
import logging
from typing import Any

from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkSessionException

logger = logging.getLogger(__name__)

_QUERY_TIMEOUT_S = 30


class SnowflakeError(Exception):
    pass


def _is_expired_token(e: Exception) -> bool:
    """The 'Authentication token has expired' error arrives as a generic exception,
    not a SnowparkSessionException — so we sniff the message."""
    s = str(e).lower()
    return "expired" in s or "authenticate again" in s or "390114" in s


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
                # Heartbeat so the session doesn't expire on idle — fixes the recurring
                # "Authentication token has expired" after the app sits idle a few hours.
                "client_session_keep_alive": True,
            }).create()
        except Exception as e:
            raise SnowflakeError(f"Failed to connect to Snowflake: {e}") from e

    def session(self) -> Session:
        if self._session is None:
            self._session = self._connect()
        return self._session

    def _run(self, sql: str, max_rows: int) -> list[dict[str, Any]]:
        rows = self.session().sql(sql).collect()
        return [r.as_dict() for r in rows[:max_rows]]

    def execute_query(self, sql: str, max_rows: int = 200) -> list[dict[str, Any]]:
        logger.info("Executing SQL: %s", sql[:200])
        try:
            return self._run(sql, max_rows)
        except SnowparkSessionException as e:
            self._session = None  # reset on session errors
            raise SnowflakeError(f"Session error: {e}") from e
        except Exception as e:
            # Expired token → drop the dead session, reconnect, and retry once (self-heal),
            # instead of failing until someone manually restarts the app.
            if _is_expired_token(e):
                logger.warning("Snowflake session expired — reconnecting and retrying once")
                self._session = None
                try:
                    return self._run(sql, max_rows)
                except Exception as e2:
                    raise SnowflakeError(f"Query failed after reconnect: {e2}") from e2
            raise SnowflakeError(f"Query failed: {e}") from e

    def get_table_schema(self, table: str) -> list[dict[str, str]]:
        rows = self.execute_query(f"DESCRIBE TABLE {table}")
        return [{"name": r.get("name", ""), "type": r.get("type", "")} for r in rows]

    def close(self):
        if self._session:
            self._session.close()
            self._session = None
