"""Unit tests for the chart sanity check. No model or database needed: `pytest -q`."""

from decimal import Decimal

from exo_agent import graph
from exo_agent.nodes.chart_decider import chart_problem
from exo_agent.nodes.executor import ExecutionResult
from exo_agent.schemas import ChartSpec, SubQuestion

# the real case from the UI: 5 planet names from this year, no numbers to plot
LATEST = [
    {"pl_name": name, "disc_year": 2026}
    for name in ["TOI-2431 b", "TOI-4336 A c", "K2-232 c", "TOI-2392 b", "TOI-4567 b"]
]
YEARLY = [{"disc_year": 2015 + i, "count": Decimal(100 + i)} for i in range(5)]


def test_bar_chart_with_no_y_axis_is_rejected():
    spec = ChartSpec(chart_type="bar", x_field="pl_name", y_field=None)
    assert chart_problem(spec, ["pl_name", "disc_year"], LATEST) == "it has no y-axis column"


def test_text_y_column_is_rejected():
    spec = ChartSpec(chart_type="bar", x_field="disc_year", y_field="pl_name")
    assert "isn't numeric" in chart_problem(spec, ["pl_name", "disc_year"], LATEST)


def test_unknown_column_and_single_row_are_rejected():
    spec = ChartSpec(chart_type="line", x_field="disc_year", y_field="num")
    assert "'num' isn't a column" in chart_problem(spec, ["disc_year", "count"], YEARLY)
    assert "only one row" in chart_problem(spec, ["disc_year", "count"], YEARLY[:1])


def test_scatter_needs_numeric_x():
    spec = ChartSpec(chart_type="scatter", x_field="pl_name", y_field="disc_year")
    assert "numeric x" in chart_problem(spec, ["pl_name", "disc_year"], LATEST)


def test_good_charts_pass():
    # Postgres COUNT comes back as Decimal from some queries; that counts as numeric
    line = ChartSpec(chart_type="line", x_field="disc_year", y_field="count")
    assert chart_problem(line, ["disc_year", "count"], YEARLY) is None
    assert chart_problem(ChartSpec(chart_type="none"), [], []) is None


def test_graph_node_drops_bad_chart_and_says_why(monkeypatch):
    bad = ChartSpec(chart_type="bar", x_field="pl_name", y_field=None)
    monkeypatch.setattr(graph, "decide_chart", lambda *args: bad)
    state = {
        "sub_question": SubQuestion(text="latest planets", needs_sql=True, needs_retrieval=False),
        "execution_result": ExecutionResult(
            success=True, sql_used="SELECT ...", columns=["pl_name", "disc_year"],
            rows=LATEST, row_count=len(LATEST),
        ),
    }
    out = graph.chart_decider_node(state)

    assert out["chart_spec"].chart_type == "none"  # so the router skips the renderer
    assert "bar chart (x = pl_name, y = None)" in out["chart_skipped"]
    assert "no y-axis" in out["chart_skipped"]
