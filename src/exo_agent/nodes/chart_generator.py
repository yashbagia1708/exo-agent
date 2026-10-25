"""
The chart-generator node: takes a ChartSpec (the decision) and the actual
query result rows, and renders a real PNG chart with matplotlib.

This node makes no LLM calls -- the decision of *what* to chart already
happened in chart_decider.py. This is pure, deterministic rendering code,
which is exactly what it should be: there's no reason to spend a model
call drawing a bar chart once we already know we want one.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")  # headless backend, no display needed -- we just save PNGs
import matplotlib.pyplot as plt

from exo_agent.schemas import ChartSpec

OUTPUT_DIR = pathlib.Path(__file__).parent.parent.parent.parent / "charts"


def generate_chart(spec: ChartSpec, rows: list[dict], filename: str = "chart.png") -> pathlib.Path | None:
    """
    Render a chart per spec from the given rows, save as PNG, return the
    path. Returns None if spec.chart_type is "none" -- nothing to render.

    Raises ValueError if chart_type isn't "none" but x_field/y_field are
    missing or aren't present in the rows (a contract violation from the
    decider that's worth surfacing loudly rather than silently skipping).
    """
    if spec.chart_type == "none":
        return None

    if not rows:
        raise ValueError("Cannot generate a chart from empty results.")

    if not spec.x_field or not spec.y_field:
        raise ValueError(
            f"chart_type='{spec.chart_type}' requires x_field and y_field, got "
            f"x_field={spec.x_field!r}, y_field={spec.y_field!r}"
        )

    available_columns = set(rows[0].keys())
    if spec.x_field not in available_columns or spec.y_field not in available_columns:
        raise ValueError(
            f"x_field/y_field must match actual result columns {sorted(available_columns)}, "
            f"got x_field={spec.x_field!r}, y_field={spec.y_field!r}"
        )

    x_values = [r[spec.x_field] for r in rows]
    y_values = [r[spec.y_field] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))

    if spec.chart_type == "line":
        ax.plot(x_values, y_values, marker="o", linewidth=1.5)
    elif spec.chart_type == "bar":
        ax.bar([str(x) for x in x_values], y_values)
        plt.xticks(rotation=30, ha="right")
    elif spec.chart_type == "scatter":
        ax.scatter(x_values, y_values, alpha=0.6)
    else:
        plt.close(fig)
        raise ValueError(f"Unknown chart_type: {spec.chart_type}")

    ax.set_xlabel(spec.x_field)
    ax.set_ylabel(spec.y_field)
    ax.set_title(spec.title or f"{spec.y_field} by {spec.x_field}")
    fig.tight_layout()

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

    return output_path


if __name__ == "__main__":
    # Manual test -- renders a real PNG from synthetic data, no DB needed
    from exo_agent.schemas import ChartSpec

    spec = ChartSpec(
        chart_type="line",
        x_field="disc_year",
        y_field="num_discovered",
        title="Exoplanet discoveries by year (test data)",
    )
    rows = [{"disc_year": y, "num_discovered": (y - 1995) * 3 % 40 + 5} for y in range(2000, 2025)]

    path = generate_chart(spec, rows, filename="test_chart.png")
    print(f"Chart saved to: {path}")
    print(f"File exists: {path.exists() if path else 'N/A'}")
    print(f"File size: {path.stat().st_size if path else 'N/A'} bytes")

    # Also test the "none" path
    none_spec = ChartSpec(chart_type="none")
    result = generate_chart(none_spec, rows)
    print(f"\n'none' chart_type returns: {result}")
