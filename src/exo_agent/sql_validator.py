"""
Lightweight, dependency-free validation of LLM-generated SQL, run *before*
anything touches Postgres.

This deliberately does NOT try to be a full SQL parser -- that's a deep
rabbit hole (see: every "is this SQL syntactically valid" library ever
written) and not the point of this project. Instead it catches the
specific, common failure modes seen in practice with small local models:

  - truncated output (a CTE or statement that never finishes)
  - obviously unbalanced parentheses/quotes
  - non-SELECT statements (safety: we never want INSERT/UPDATE/DELETE/DROP)
  - empty or near-empty output

Anything that passes these checks still goes through real execution next,
where Postgres's own parser is the actual source of truth. This is a fast
first filter, not a guarantee of correctness.
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

    # Must start with SELECT or WITH (for CTEs that end in a SELECT)
    first_word = stripped.split(None, 1)[0].upper()
    if first_word not in ("SELECT", "WITH"):
        return ValidationResult(
            False, f"Query must start with SELECT or WITH, got '{first_word}'."
        )

    # Forbidden statement types anywhere in the query (defense in depth --
    # even inside a CTE or subquery, these should never appear)
    upper = stripped.upper()
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            return ValidationResult(False, f"Query contains forbidden keyword '{kw}'.")

    # Balanced parentheses -- catches truncated CTEs/subqueries
    if stripped.count("(") != stripped.count(")"):
        return ValidationResult(
            False,
            f"Unbalanced parentheses: {stripped.count('(')} open vs "
            f"{stripped.count(')')} close. The query is likely incomplete.",
        )

    # Balanced single quotes (rough check -- doesn't handle escaped quotes,
    # but catches the common truncation case)
    if stripped.count("'") % 2 != 0:
        return ValidationResult(False, "Unbalanced single quotes; query is likely incomplete.")

    # A WITH (CTE) query must contain a final SELECT after its CTE definitions,
    # not just trail off after the last comma-separated CTE.
    if first_word == "WITH":
        # crude but effective: a CTE block trailing in a comma with nothing
        # meaningful after it is the exact truncation pattern we saw in testing
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
    # Manual smoke tests, including the exact truncated query we saw for real
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
