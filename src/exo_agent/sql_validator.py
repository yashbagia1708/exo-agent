"""
Cheap checks on model-written SQL before it reaches Postgres.

Not a parser. Catches what small local models actually get wrong:
empty output, truncated CTEs, unbalanced brackets/quotes, and anything
that isn't a read-only SELECT. Postgres still has the final say.
"""

import re
from dataclasses import dataclass

FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "GRANT", "REVOKE", "EXEC", "EXECUTE",
]


@dataclass
class ValidationResult:
    is_valid: bool
    error: str | None = None


def validate_sql(sql: str) -> ValidationResult:
    stripped = sql.strip().rstrip(";").strip()

    if not stripped:
        return ValidationResult(False, "SQL is empty.")

    # read-only: SELECT, or WITH ... SELECT
    first_word = stripped.split(None, 1)[0].upper()
    if first_word not in ("SELECT", "WITH"):
        return ValidationResult(
            False, f"Query must start with SELECT or WITH, got '{first_word}'."
        )

    # no writes anywhere, including inside a CTE (whole words only: 'created' is fine)
    upper = stripped.upper()
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            return ValidationResult(False, f"Query contains forbidden keyword '{kw}'.")

    # unbalanced brackets usually means the model stopped mid-query
    if stripped.count("(") != stripped.count(")"):
        return ValidationResult(
            False,
            f"Unbalanced parentheses: {stripped.count('(')} open vs "
            f"{stripped.count(')')} close. The query is likely incomplete.",
        )

    # rough: ignores escaped quotes, but catches a cut-off string literal
    if stripped.count("'") % 2 != 0:
        return ValidationResult(False, "Unbalanced single quotes; query is likely incomplete.")

    # WITH a AS (...), b AS (...),   <- seen for real: CTEs with no final SELECT
    if first_word == "WITH":
        if stripped.rstrip().endswith(","):
            return ValidationResult(
                False,
                "Query ends with a trailing comma after a CTE definition -- "
                "it looks truncated before a final SELECT statement.",
            )
        if "SELECT" not in upper.split("AS", 1)[-1] and upper.count("SELECT") < 2:
            return ValidationResult(
                False,
                "A WITH query needs at least one SELECT inside the CTE and "
                "one final SELECT using it -- this query appears to be missing "
                "the final SELECT.",
            )

    return ValidationResult(True)


if __name__ == "__main__":
    # quick manual check; the second query is the real truncated one
    test_cases = [
        "SELECT COUNT(*) FROM planets;",
        "WITH yearly_discoveries AS (SELECT hostname, disc_year, COUNT(*) as num_discovered FROM planets GROUP BY hostname, disc_year),",
        "SELECT * FROM planets WHERE pl_name = 'Kepler-80 d';",
        "DROP TABLE planets;",
        "",
        "WITH x AS (SELECT 1) SELECT * FROM x;",
    ]
    for sql in test_cases:
        result = validate_sql(sql)
        status = "VALID" if result.is_valid else f"INVALID ({result.error})"
        print(f"[{status}]\n  {sql!r}\n")
