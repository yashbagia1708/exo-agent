"""Unit tests for the exact discovery-method rewrite. No database: `pytest -q`."""

from exo_agent.nodes import executor
from exo_agent.nodes.executor import exact_method_filter

LABELS = ("Transit", "Radial Velocity", "Transit Timing Variations", "Imaging")


def test_substring_on_an_exact_label_becomes_equals():
    # the real query from the tier 4 eval: counted 4,814 instead of 4,785
    sql = "SELECT COUNT(*) FROM planets WHERE discoverymethod ILIKE '%Transit%'"
    assert exact_method_filter(sql, LABELS) == "SELECT COUNT(*) FROM planets WHERE discoverymethod = 'Transit'"


def test_case_and_alias_are_handled():
    sql = "SELECT * FROM planets p WHERE p.discoverymethod like '%radial velocity%'"
    assert exact_method_filter(sql, LABELS) == "SELECT * FROM planets p WHERE p.discoverymethod = 'Radial Velocity'"


def test_partial_matches_are_left_alone():
    sql = "SELECT * FROM planets WHERE discoverymethod ILIKE '%Timing%'"
    assert exact_method_filter(sql, LABELS) == sql


def test_other_columns_and_exact_filters_are_untouched():
    for sql in [
        "SELECT * FROM planets WHERE disc_facility ILIKE '%TESS%'",
        "SELECT * FROM planets WHERE discoverymethod = 'Transit'",
        "SELECT * FROM planets WHERE pl_name ILIKE '%Transit%'",
    ]:
        assert exact_method_filter(sql, LABELS) == sql


def test_no_database_call_without_a_method_filter(monkeypatch):
    def boom():
        raise AssertionError("labels shouldn't be looked up")

    monkeypatch.setattr(executor, "_discovery_methods", boom)
    assert exact_method_filter("SELECT COUNT(*) FROM planets") == "SELECT COUNT(*) FROM planets"


def test_executor_runs_the_tightened_query(monkeypatch):
    ran = []
    monkeypatch.setattr(executor, "_discovery_methods", lambda: LABELS)
    monkeypatch.setattr(executor, "_run_query", lambda sql: ran.append(sql) or (["count"], [{"count": 4785}]))

    result = executor.execute_sql("SELECT COUNT(*) FROM planets WHERE discoverymethod ILIKE '%Transit%'", "q")

    assert ran == ["SELECT COUNT(*) FROM planets WHERE discoverymethod = 'Transit'"]
    assert result.rows == [{"count": 4785}] and result.repair_attempts == 0
