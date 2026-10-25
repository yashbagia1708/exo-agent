"""
Chart decider: look at the actual result, then pick none/line/bar/scatter.

The SQL writer guessed needs_chart from the question alone. This runs
after the query, so it can see the real shape, e.g. a "trend" question
that came back as one row gets no chart.
"""

from exo_agent import config
from exo_agent.providers.factory import get_client
from exo_agent.schemas import ChartSpec

SYSTEM_PROMPT = """You decide whether query results should be shown as a \
chart, and if so, what kind. You will be given the original question and \
a sample of the actual query results (column names, row count, and a few \
sample rows).

Rules:
- A single row or single value (e.g. a COUNT, an average, "the largest X")
  should almost never be a chart -- chart_type "none" is correct here.
- A trend over time (multiple rows, one column is a year/date, another is
  a count or measurement) is a good fit for "line".
- A comparison across a small number of categories (e.g. counts grouped by
  discovery method) is a good fit for "bar".
- A relationship between two continuous numeric variables across many rows
  (e.g. stellar temperature vs planet size) is a good fit for "scatter".
- If there are more than ~50 rows and no clear x/y structure, or the data
  doesn't fit any of the above patterns cleanly, use "none" rather than
  forcing a bad chart.
- x_field and y_field must be exact column names from the provided results.
"""

USER_PROMPT = """Original question: "{question}"

Result columns: {columns}
Row count: {row_count}
Sample rows: {sample_rows}

Decide the chart type and field mapping.
"""


def decide_chart(question: str, columns: list[str], rows: list[dict]) -> ChartSpec:
    """Always returns a ChartSpec ("none" if no chart). 5 sample rows is enough to judge shape."""
    client = get_client()

    result = client.chat.completions.create(
        model=config.MODEL,
        response_model=ChartSpec,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    question=question,
                    columns=columns,
                    row_count=len(rows),
                    sample_rows=rows[:5],
                ),
            },
        ],
    )
    return result


if __name__ == "__main__":
    # synthetic shapes, no DB needed: trend -> line, single count -> none, categories -> bar
    cases = [
        {
            "question": "How has the rate of new exoplanet discoveries changed by year since 2000?",
            "columns": ["disc_year", "num_discovered"],
            "rows": [{"disc_year": y, "num_discovered": y % 20 + 5} for y in range(2000, 2025)],
        },
        {
            "question": "How many confirmed exoplanets are there?",
            "columns": ["count"],
            "rows": [{"count": 6298}],
        },
        {
            "question": "How many planets were found by each discovery method?",
            "columns": ["discoverymethod", "count"],
            "rows": [
                {"discoverymethod": "Transit", "count": 4500},
                {"discoverymethod": "Radial Velocity", "count": 1100},
                {"discoverymethod": "Imaging", "count": 70},
            ],
        },
    ]

    for case in cases:
        print(f"\nQuestion: {case['question']}")
        print(f"Shape: {len(case['rows'])} rows, columns={case['columns']}")
        spec = decide_chart(case["question"], case["columns"], case["rows"])
        print(f"  -> chart_type={spec.chart_type}, x={spec.x_field}, y={spec.y_field}, title={spec.title}")
