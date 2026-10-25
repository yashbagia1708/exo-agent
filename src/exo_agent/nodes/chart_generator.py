"""
Chart generator: ChartSpec + rows -> PNG in charts/. Plain matplotlib, no LLM.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")  # no display; we only save files
import matplotlib.pyplot as plt

from exo_agent.schemas import ChartSpec

OUTPUT_DIR = pathlib.Path(__file__).parent.parent.parent.parent / "charts"


def generate_chart(spec: ChartSpec, rows: list[dict], filename: str = "chart.png") -> pathlib.Path | None:
    """
    Save the chart and return its path, or None for chart_type "none".
    Raises ValueError if the decider named axes that aren't in the rows.
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
    # renders charts/test_chart.png from fake data, no DB needed
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

    # "none" should render nothing
    none_spec = ChartSpec(chart_type="none")
    result = generate_chart(none_spec, rows)
    print(f"\n'none' chart_type returns: {result}")
