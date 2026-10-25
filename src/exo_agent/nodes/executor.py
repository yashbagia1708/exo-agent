"""
The executor node: runs a validated SQLQuery against the real Postgres
database and returns results, OR repairs the query if Postgres itself
rejects it.

This is the second half of the project's repair loop. sql_validator.py
(used in sql_writer.py) catches structural malformation -- truncated
CTEs, unbalanced parens -- before the database is ever touched. This
node catches everything else: semantic errors (e.g. EXTRACT(YEAR FROM
an_integer_column), a type mismatch that's structurally valid SQL but
fails at execution), nonexistent columns, bad joins, and so on.
Postgres's own error message is the ground truth here -- it's fed
straight back to the model, the same retry pattern as the structural
repair loop.
"""

from dataclasses import dataclass, field

import psycopg2
import psycopg2.extras

from exo_agent import config
from exo_agent.providers.factory import get_client
from exo_agent.schema_context import get_schema_context
from exo_agent.schemas import SQLRepair

MAX_ROWS_RETURNED = 200  # safety cap so a broad query can't flood the agent's context

REPAIR_SYSTEM_PROMPT = """You are a careful SQL analyst working with a PostgreSQL \
database of confirmed exoplanets. Your previous query failed when actually \
run against the database. Fix it using the real error message and the \
schema below. Write a single, complete, correct SELECT statement.

{schema}
"""

REPAIR_USER_PROMPT = """Original question: "{question}"

Query that failed:
  {sql}

Postgres error:
  {error}

Write a corrected SELECT statement that fixes this specific error and \
still answers the original question.
"""


@dataclass
class ExecutionResult:
    success: bool
    sql_used: str
    columns: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    error: str | None = None
    repair_attempts: int = 0


def _run_query(sql: str) -> tuple[list[str], list[dict]]:
    """Execute SQL and return (column_names, rows_as_dicts). Raises psycopg2 errors as-is."""
    conn = psycopg2.connect(config.DB_DSN)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchmany(MAX_ROWS_RETURNED + 1)
            columns = [desc[0] for desc in cur.description] if cur.description else []
            return columns, [dict(r) for r in rows]
    finally:
        conn.close()


def execute_sql(sql: str, question: str, max_repair_attempts: int | None = None) -> ExecutionResult:
    """
    Execute sql. If Postgres raises an error, feed that error back to the
    model and retry, up to max_repair_attempts times (default:
    config.MAX_SQL_REPAIR_ATTEMPTS).

    Always returns an ExecutionResult -- never raises on a SQL execution
    failure. Only raises on infrastructure problems (e.g. can't connect to
    Postgres at all), since that's not something a repair retry can fix.
    """
    if max_repair_attempts is None:
        max_repair_attempts = config.MAX_SQL_REPAIR_ATTEMPTS

    current_sql = sql
    client = None  # lazily created only if we actually need a repair

    for attempt in range(max_repair_attempts + 1):
        try:
            columns, rows = _run_query(current_sql)
            truncated = len(rows) > MAX_ROWS_RETURNED
            if truncated:
                rows = rows[:MAX_ROWS_RETURNED]
            return ExecutionResult(
                success=True,
                sql_used=current_sql,
                columns=columns,
                rows=rows,
                row_count=len(rows),
                truncated=truncated,
                repair_attempts=attempt,
            )
        except psycopg2.Error as e:
            error_msg = str(e).strip()
            print(f"  [executor] Attempt {attempt + 1} failed: {error_msg}")

            if attempt >= max_repair_attempts:
                return ExecutionResult(
                    success=False,
                    sql_used=current_sql,
                    error=error_msg,
                    repair_attempts=attempt,
                )

            if client is None:
                client = get_client()

            repair: SQLRepair = client.chat.completions.create(
                model=config.MODEL,
                response_model=SQLRepair,
                messages=[
                    {
                        "role": "system",
                        "content": REPAIR_SYSTEM_PROMPT.format(schema=get_schema_context()),
                    },
                    {
                        "role": "user",
                        "content": REPAIR_USER_PROMPT.format(
                            question=question, sql=current_sql, error=error_msg
                        ),
                    },
                ],
            )
            print(f"  [executor] Repair attempt: {repair.what_was_wrong}")
            current_sql = repair.sql

    # Unreachable, but keeps type-checkers happy
    return ExecutionResult(success=False, sql_used=current_sql, error="Unknown failure.")


if __name__ == "__main__":
    # Manual smoke test exercising the repair loop against a query that's
    # structurally valid but fails at execution time (EXTRACT on an
    # integer column):
    #   python3 -m exo_agent.nodes.executor
    bad_sql = (
        "SELECT EXTRACT(YEAR FROM disc_year) as discover_year, COUNT(pl_name) as num_discovered "
        "FROM planets GROUP BY disc_year"
    )
    question = "How has the rate of new exoplanet discoveries changed by year since 2000?"

    print(f"Testing executor repair loop with known-bad SQL:\n  {bad_sql}\n")
    result = execute_sql(bad_sql, question)

    if result.success:
        print(f"\nSucceeded after {result.repair_attempts} repair attempt(s).")
        print(f"Final SQL: {result.sql_used}")
        print(f"Columns: {result.columns}")
        print(f"Row count: {result.row_count}")
        print("First few rows:")
        for row in result.rows[:5]:
            print(f"  {row}")
    else:
        print(f"\nFailed after {result.repair_attempts} repair attempt(s).")
        print(f"Last error: {result.error}")
