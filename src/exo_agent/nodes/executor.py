"""
Executor: run the SQL, and if Postgres rejects it, ask the model to fix it.

sql_writer already caught structural problems (truncation, brackets).
This catches what only the database can see:
  - errors: wrong types (EXTRACT(YEAR FROM disc_year) on an integer),
    missing columns, bad joins -> Postgres's message goes back to the model
  - 0 rows from a filter on the wrong column (discoverymethod ILIKE
    '%TESS%') -> we look the value up and send back where it really is
"""

import re
from dataclasses import dataclass, field

import psycopg2
import psycopg2.extras

from exo_agent import config
from exo_agent.providers.factory import get_client
from exo_agent.schema_context import get_schema_context
from exo_agent.schemas import SQLRepair

MAX_ROWS_RETURNED = 200  # cap so a broad query doesn't flood the prompt

# text columns a filter value might really belong to
TEXT_COLUMNS = [
    ("planets", "pl_name"),
    ("planets", "hostname"),
    ("planets", "discoverymethod"),
    ("planets", "disc_facility"),
    ("host_stars", "st_spectype"),
]
# col = 'x' / col ILIKE '%x%' / p.col LIKE 'x'  ->  (col, op, 'x')
TEXT_FILTER = re.compile(r"(?:\w+\.)?(\w+)\s+(=|I?LIKE)\s+'((?:[^']|'')*)'", re.IGNORECASE)

REPAIR_SYSTEM_PROMPT = """You are a careful SQL analyst working with a PostgreSQL \
database of confirmed exoplanets. Your previous query failed when actually \
run against the database. Fix it using the real error message and the \
schema below. Write a single, complete, correct SELECT statement.

{schema}
"""

REPAIR_USER_PROMPT = """Original question: "{question}"

Query that failed:
  {sql}

What went wrong:
  {error}

Write a corrected SELECT statement that fixes this specific problem and \
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
    repair_notes: list[str] = field(default_factory=list)  # what each repair was fixing
    zero_row_note: str | None = None  # set if it still came back empty after the filter check


def _run_query(sql: str) -> tuple[list[str], list[dict]]:
    """-> (column names, rows as dicts). psycopg2 errors bubble up."""
    conn = psycopg2.connect(config.DB_DSN)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchmany(MAX_ROWS_RETURNED + 1)
            columns = [desc[0] for desc in cur.description] if cur.description else []
            return columns, [dict(r) for r in rows]
    finally:
        conn.close()


def nulls_last(sql: str) -> str:
    """
    'ORDER BY pl_rade DESC' -> 'ORDER BY pl_rade DESC NULLS LAST'

    Postgres sorts NULL first on DESC, so "largest planet" came back as one
    with no measured radius (FINDINGS.md, 2026-06-22 and 2026-10-26). The
    prompt warns about this but the model doesn't always listen. Skips text
    inside quotes and any DESC that already says NULLS.
    """
    parts = re.split(r"('(?:[^']|'')*')", sql)  # odd indexes are string literals
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"\bDESC\b(?!\s+NULLS)", lambda m: m.group(0) + " NULLS LAST", parts[i], flags=re.IGNORECASE)
    return "".join(parts)


def _find_value_in_columns(value: str) -> list[tuple[str, str, int, str]]:
    """'TESS' -> [('planets', 'disc_facility', 1012, 'Transiting Exoplanet Survey Satellite (TESS)')]"""
    hits = []
    conn = psycopg2.connect(config.DB_DSN)
    try:
        with conn.cursor() as cur:
            for table, column in TEXT_COLUMNS:
                cur.execute(
                    f"SELECT count(*), min({column}) FROM {table} WHERE {column} ILIKE %s",
                    (f"%{value}%",),
                )
                count, example = cur.fetchone()
                if count:
                    hits.append((table, column, count, example))
    finally:
        conn.close()
    return hits


def empty_result_hint(sql: str, find=None) -> str | None:
    """
    A query returned 0 rows. Did a text filter look in the wrong place?

        discoverymethod ILIKE '%TESS%' -> "TESS" is in disc_facility (1012 rows)
        disc_facility = 'TESS'         -> no exact match, but 1012 rows contain it
        pl_name ILIKE '%Zorblax%'      -> None: it's nowhere, the empty result is real
    """
    find = find or _find_value_in_columns
    hints = []
    for column, op, raw in TEXT_FILTER.findall(sql):
        column, value = column.lower(), raw.replace("''", "'").strip("%").strip()
        if len(value) < 2:
            continue
        already_substring = op.upper() == "ILIKE" and raw.startswith("%") and raw.endswith("%")
        for table, other, count, example in sorted(find(value), key=lambda h: -h[2]):
            if other != column:
                hints.append(
                    f"The filter {column} {op} '{raw}' matched nothing, but '{value}' appears in "
                    f"{table}.{other} ({count} rows, e.g. '{example}'). Filter on {other} instead."
                )
                break
            if not already_substring:
                hints.append(
                    f"No row has {column} {op} '{raw}', but {count} rows contain '{value}' "
                    f"(e.g. '{example}'). Use {column} ILIKE '%{value}%'."
                )
                break
    return ("0 rows: " + " ".join(hints)) if hints else None


def execute_sql(sql: str, question: str, max_repair_attempts: int | None = None) -> ExecutionResult:
    """
    Run sql and repair it (up to MAX_SQL_REPAIR_ATTEMPTS) when:
      - Postgres raises an error, or
      - it returns 0 rows and a text filter clearly looked in the wrong place
        (see empty_result_hint), so we don't report "no data" for a bad filter
    A bad query returns success=False rather than raising. Only connection
    problems raise, since a retry can't fix those.
    """
    if max_repair_attempts is None:
        max_repair_attempts = config.MAX_SQL_REPAIR_ATTEMPTS

    current_sql = sql
    client = None  # only needed if a repair happens
    repair_notes: list[str] = []

    for attempt in range(max_repair_attempts + 1):
        current_sql = nulls_last(current_sql)
        try:
            columns, rows = _run_query(current_sql)
        except psycopg2.Error as e:
            problem = str(e).strip()
            print(f"  [executor] Attempt {attempt + 1} failed: {problem}")
        else:
            hint = None if rows else empty_result_hint(current_sql)
            if not hint or attempt >= max_repair_attempts:
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
                    repair_notes=repair_notes,
                    zero_row_note=hint,
                )
            problem = hint
            print(f"  [executor] {hint}")

        if attempt >= max_repair_attempts:
            return ExecutionResult(
                success=False,
                sql_used=current_sql,
                error=problem,
                repair_attempts=attempt,
                repair_notes=repair_notes,
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
                        question=question, sql=current_sql, error=problem
                    ),
                },
            ],
        )
        print(f"  [executor] Repair attempt: {repair.what_was_wrong}")
        repair_notes.append(problem)
        current_sql = repair.sql

    # unreachable; keeps type checkers happy
    return ExecutionResult(success=False, sql_used=current_sql, error="Unknown failure.")


if __name__ == "__main__":
    # python3 -m exo_agent.nodes.executor
    # valid-looking SQL that Postgres rejects (EXTRACT on an integer) -> should self-repair
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
