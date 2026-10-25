"""Unit tests for the 0-row filter check. Fake database and model: `pytest -q`."""

from types import SimpleNamespace

from exo_agent.nodes import executor
from exo_agent.nodes.executor import empty_result_hint
from exo_agent.schemas import SQLRepair

TESS = "Transiting Exoplanet Survey Satellite (TESS)"


def fake_find(value):
    """Pretend database: 'TESS' only lives in disc_facility."""
    return [("planets", "disc_facility", 1012, TESS)] if value.upper() == "TESS" else []


def test_wrong_column_is_spotted():
    # the real failing query from the live run
    sql = "SELECT pl_name, pl_rade FROM planets WHERE discoverymethod ILIKE '%TESS%' ORDER BY pl_rade DESC"
    hint = empty_result_hint(sql, find=fake_find)
    assert hint.startswith("0 rows:")
    assert "disc_facility" in hint and "Filter on disc_facility instead" in hint


def test_exact_match_on_a_longer_value_suggests_ilike():
    hint = empty_result_hint("SELECT * FROM planets WHERE p.disc_facility = 'TESS'", find=fake_find)
    assert "Use disc_facility ILIKE '%TESS%'" in hint


def test_value_that_exists_nowhere_is_a_real_empty_result():
    assert empty_result_hint("SELECT * FROM planets WHERE pl_name ILIKE '%Zorblax%'", find=fake_find) is None


def test_numeric_filters_are_left_alone():
    assert empty_result_hint("SELECT * FROM planets WHERE disc_year = 2030", find=fake_find) is None


def test_executor_retries_with_the_hint_then_returns_rows(monkeypatch):
    bad = "SELECT pl_name FROM planets WHERE discoverymethod ILIKE '%TESS%'"
    good = "SELECT pl_name FROM planets WHERE disc_facility ILIKE '%TESS%'"
    rows = {bad: [], good: [{"pl_name": "TOI-1408 b"}]}
    monkeypatch.setattr(executor, "_run_query", lambda sql: (["pl_name"], rows[sql]))
    monkeypatch.setattr(executor, "_find_value_in_columns", fake_find)

    sent = []

    def create(**kwargs):
        sent.append(kwargs["messages"][-1]["content"])
        return SQLRepair(sql=good, what_was_wrong="Filtered the wrong column.")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(executor, "get_client", lambda: client)

    result = executor.execute_sql(bad, "largest planet found by TESS?")

    assert result.success and result.rows == [{"pl_name": "TOI-1408 b"}]
    assert result.repair_attempts == 1 and result.sql_used == good
    assert result.repair_notes[0].startswith("0 rows:")
    assert "disc_facility" in sent[0]  # the model was told where the value really is


def test_real_empty_result_makes_no_model_call(monkeypatch):
    monkeypatch.setattr(executor, "_run_query", lambda sql: (["pl_name"], []))
    monkeypatch.setattr(executor, "_find_value_in_columns", fake_find)
    monkeypatch.setattr(executor, "get_client", lambda: (_ for _ in ()).throw(AssertionError("no repair expected")))

    result = executor.execute_sql("SELECT pl_name FROM planets WHERE pl_name ILIKE '%Zorblax%'", "q")
    assert result.success and result.row_count == 0 and result.repair_attempts == 0


def test_desc_sorts_put_nulls_last():
    from exo_agent.nodes.executor import nulls_last

    assert nulls_last("SELECT pl_name FROM planets ORDER BY pl_rade DESC LIMIT 1") == (
        "SELECT pl_name FROM planets ORDER BY pl_rade DESC NULLS LAST LIMIT 1"
    )
    assert nulls_last("ORDER BY a desc, b ASC, c DESC NULLS FIRST") == (
        "ORDER BY a desc NULLS LAST, b ASC, c DESC NULLS FIRST"
    )
    assert nulls_last("SELECT 'DESC' AS x ORDER BY y") == "SELECT 'DESC' AS x ORDER BY y"
