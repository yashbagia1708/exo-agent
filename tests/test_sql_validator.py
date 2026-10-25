"""Unit tests for sql_validator. No model or database needed: `pytest -q`."""

import pytest

from exo_agent.sql_validator import validate_sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM planets;",
        "  select pl_name from planets where disc_year = 2016  ",
        "SELECT * FROM planets WHERE pl_name = 'Kepler-80 d';",
        "WITH x AS (SELECT 1) SELECT * FROM x;",
        "SELECT p.pl_name, h.st_teff FROM planets p JOIN host_stars h USING (hostname)",
    ],
)
def test_accepts_read_only_queries(sql):
    result = validate_sql(sql)
    assert result.is_valid, result.error


@pytest.mark.parametrize(
    "sql, expected_error",
    [
        ("", "empty"),
        ("   ;  ", "empty"),
        ("DROP TABLE planets;", "must start with SELECT or WITH"),
        ("SELECT 1; DELETE FROM planets", "forbidden keyword 'DELETE'"),
        ("WITH x AS (DELETE FROM planets RETURNING *) SELECT * FROM x", "forbidden keyword 'DELETE'"),
        ("SELECT COUNT(* FROM planets", "Unbalanced parentheses"),
        ("SELECT * FROM planets WHERE pl_name = 'Kepler-80 d", "Unbalanced single quotes"),
    ],
)
def test_rejects_unsafe_or_malformed_queries(sql, expected_error):
    result = validate_sql(sql)
    assert not result.is_valid
    assert expected_error in result.error


def test_rejects_truncated_cte_seen_in_practice():
    sql = (
        "WITH yearly_discoveries AS (SELECT hostname, disc_year, COUNT(*) as "
        "num_discovered FROM planets GROUP BY hostname, disc_year),"
    )
    result = validate_sql(sql)
    assert not result.is_valid
    assert "trailing comma" in result.error


def test_forbidden_keyword_must_be_a_whole_word():
    # 'created' contains CREATE, 'updated_at' contains UPDATE; both are fine
    result = validate_sql("SELECT pl_name AS created, disc_year AS updated_at FROM planets")
    assert result.is_valid, result.error
