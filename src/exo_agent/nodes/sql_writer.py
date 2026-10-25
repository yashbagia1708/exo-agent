"""
The SQL-writer node: takes a natural-language question and produces a
single, schema-grounded, validated SELECT statement.

This node never invents column names -- it's given the real schema as
context (schema_context.py) and constrained to return a validated SQLQuery
object via instructor, so the LLM cannot return prose instead of a query.

It also runs every generated query through a lightweight syntax validator
(sql_validator.py) before returning it. If validation fails -- e.g. a
truncated CTE, unbalanced parens, a forbidden statement type -- the error
is fed back to the model and it gets another attempt, up to
config.MAX_SQL_REPAIR_ATTEMPTS times. This is the first half of the
project's tool-call repair loop; the second half (handling real Postgres
execution errors) lives in the executor node.

Before writing SQL, this node also calls entity_resolver.py: if the
question mentions a specific named planet or star, that name is resolved
against the real database via deterministic trigram similarity rather
than left for the model to guess. Catalogue names are often abbreviated
or formatted differently than common usage, so handing the model an
exact-match instruction alone isn't enough -- it still has to choose the
right substring to search for. Resolving the name first and passing the
confirmed value directly removes that guess entirely.
"""

from exo_agent import config
from exo_agent.nodes.entity_resolver import resolve_entity
from exo_agent.providers.factory import get_client
from exo_agent.schema_context import get_schema_context
from exo_agent.schemas import SQLQuery
from exo_agent.sql_validator import validate_sql

SYSTEM_PROMPT = """You are a careful SQL analyst working with a PostgreSQL \
database of confirmed exoplanets. You write a single, correct, COMPLETE SELECT \
statement that answers the user's question, using only the tables and \
columns described below. You never guess at columns that aren't listed. \
If a question asks for something not present in the schema (e.g. surface \
temperature, confirmed habitability, presence of life), write a query that \
gets the closest real proxy available and note that limitation in your \
reasoning -- do not silently invent data.

If you use a WITH clause (CTE), you must always finish it with a complete \
final SELECT statement that uses it. Never leave a query trailing off after \
a CTE definition.

{entity_context}

{schema}
"""

ENTITY_CONTEXT_RESOLVED = """CONFIRMED ENTITY: the question mentions "{mentioned_as}", \
which has been looked up against the real database and confirmed to match \
exactly: {resolved_name} (found in the {table} table, similarity={score:.2f}). \
Use this EXACT value in an equality filter (e.g. WHERE pl_name = '{resolved_name}' \
or WHERE hostname = '{resolved_name}') -- do not use ILIKE or guess at a \
different spelling, since this name has already been verified against the \
real data.
"""

ENTITY_CONTEXT_UNRESOLVED = """NOTE: the question mentions "{mentioned_as}", but no \
confident match was found in the database (best similarity was below the \
matching threshold). This object likely does NOT exist in this dataset under \
any name variant. Write a query that would return zero rows for it (e.g. an \
ILIKE search on a short distinctive substring), and let the empty result \
speak for itself -- do not assume it exists under some other spelling.
"""

REPAIR_PROMPT = """Your previous SQL query failed validation:

  {sql}

Validation error: {error}

Write a corrected, COMPLETE SELECT statement that fixes this specific \
problem and still answers the original question: "{question}"
"""


def _build_entity_context_text(resolved) -> str:
    """Format an already-resolved entity (or None) into prompt text."""
    if resolved is None:
        return ""

    if resolved.resolved_name is not None:
        return ENTITY_CONTEXT_RESOLVED.format(
            mentioned_as=resolved.mentioned_as,
            resolved_name=resolved.resolved_name,
            table=resolved.table,
            score=resolved.similarity_score,
        )

    return ENTITY_CONTEXT_UNRESOLVED.format(mentioned_as=resolved.mentioned_as)


def write_sql(question: str, max_attempts: int | None = None, resolved_entity=None) -> SQLQuery:
    """
    Given a natural-language question, return a validated SQLQuery whose
    .sql field has passed syntax validation (not yet real execution --
    that happens in the executor node).

    resolved_entity: an already-computed ResolvedEntity (from
    entity_resolver.resolve_entity()), or None. If not provided, this
    function resolves it itself -- callers that need to SEE the
    resolution result separately (e.g. graph.py, to show it in a trace)
    should call resolve_entity() themselves first and pass it in here,
    so the lookup isn't run twice.

    Retries up to max_attempts times (default: config.MAX_SQL_REPAIR_ATTEMPTS)
    if generated SQL fails validation, feeding the validator's error back
    to the model each time.

    Raises RuntimeError if the model can't produce valid SQL within the
    attempt budget. Raises whatever the underlying client raises on a
    hard failure (e.g. the provider being unreachable).
    """
    if max_attempts is None:
        max_attempts = config.MAX_SQL_REPAIR_ATTEMPTS + 1  # +1 for the initial attempt

    client = get_client()
    schema = get_schema_context()

    if resolved_entity is None:
        try:
            resolved_entity = resolve_entity(question)
        except Exception as e:
            print(f"  [sql_writer] Entity resolution failed ({e}), proceeding without it.")
            resolved_entity = None

    if resolved_entity is not None:
        if resolved_entity.resolved_name:
            print(f"  [sql_writer] Resolved '{resolved_entity.mentioned_as}' -> "
                  f"'{resolved_entity.resolved_name}' (similarity={resolved_entity.similarity_score:.2f})")
        else:
            print(f"  [sql_writer] Could not confidently resolve '{resolved_entity.mentioned_as}' "
                  f"(best similarity={resolved_entity.similarity_score})")

    entity_context = _build_entity_context_text(resolved_entity)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(entity_context=entity_context, schema=schema),
        },
        {"role": "user", "content": question},
    ]

    last_result = None
    last_error = None

    for attempt in range(1, max_attempts + 1):
        result = client.chat.completions.create(
            model=config.MODEL,
            response_model=SQLQuery,
            messages=messages,
        )
        last_result = result

        validation = validate_sql(result.sql)
        if validation.is_valid:
            if attempt > 1:
                print(f"  [sql_writer] Fixed after {attempt} attempt(s).")
            return result

        last_error = validation.error
        print(f"  [sql_writer] Attempt {attempt} failed validation: {validation.error}")

        if attempt < max_attempts:
            messages.append({"role": "assistant", "content": result.sql})
            messages.append(
                {
                    "role": "user",
                    "content": REPAIR_PROMPT.format(
                        sql=result.sql, error=validation.error, question=question
                    ),
                }
            )

    raise RuntimeError(
        f"Could not produce valid SQL after {max_attempts} attempts. "
        f"Last error: {last_error}. Last SQL: {last_result.sql if last_result else 'N/A'}"
    )


if __name__ == "__main__":
    # Quick manual test -- not part of the eval suite, just a sanity check
    # you can run directly: python3 -m exo_agent.nodes.sql_writer "<question>"
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "How many confirmed exoplanets are there?"
    print(f"Question: {question}\n")
    result = write_sql(question)
    print(f"\nSQL:\n  {result.sql}\n")
    print(f"Reasoning: {result.reasoning}")
    print(f"Needs chart: {result.needs_chart}")
