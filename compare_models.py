"""
Quick, manual comparison of how different local models handle the same
known-tricky questions -- cases that exercise edge conditions like
date-range filters, multi-step CTEs, and type coercion in generated SQL.

This is NOT the formal eval harness (that's eval/run_eval.py, run against
the full question set). This is a fast, deliberately narrow script for
comparing a couple of local models side by side on a small set of
representative questions while you have multiple models pulled and warm.

Usage:
    python3 compare_models.py llama3.1:8b qwen2.5-coder:14b qwen2.5-coder:32b
"""

import sys
import time

from exo_agent import config
from exo_agent.nodes.executor import execute_sql
from exo_agent.nodes.sql_writer import write_sql

TEST_QUESTIONS = [
    {
        "id": "trend_since_2000",
        "question": "How has the rate of new exoplanet discoveries changed by year since 2000?",
        "watch_for": "Does it include a 'since 2000' filter (disc_year >= 2000)? "
                     "Does it avoid EXTRACT(YEAR FROM disc_year) since disc_year is already an integer?",
    },
    {
        "id": "surface_temp_proxy",
        "question": "Which planet has the highest surface temperature?",
        "watch_for": "Does it use pl_eqt as a named proxy and flag the limitation, "
                     "rather than silently treating pl_eqt as literal surface temperature?",
    },
]


def run_for_model(model_name: str):
    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 70}")

    # Temporarily override the configured model for this run
    config.MODEL = model_name

    for case in TEST_QUESTIONS:
        print(f"\n--- {case['id']} ---")
        print(f"Question: {case['question']}")
        print(f"Watch for: {case['watch_for']}\n")

        start = time.time()
        try:
            sql_result = write_sql(case["question"])
        except Exception as e:
            print(f"  FAILED at write_sql: {e}")
            continue
        write_elapsed = time.time() - start

        print(f"SQL ({write_elapsed:.1f}s to generate):\n  {sql_result.sql}")
        print(f"Needs chart: {sql_result.needs_chart}")

        start = time.time()
        exec_result = execute_sql(sql_result.sql, case["question"])
        exec_elapsed = time.time() - start

        if exec_result.success:
            print(f"\n  EXECUTION SUCCEEDED ({exec_elapsed:.1f}s, "
                  f"{exec_result.repair_attempts} repair attempt(s))")
            print(f"  Final SQL: {exec_result.sql_used}")
            print(f"  Rows returned: {exec_result.row_count}")
            print(f"  Sample: {exec_result.rows[:3]}")
        else:
            print(f"\n  EXECUTION FAILED after {exec_result.repair_attempts} repair attempt(s)")
            print(f"  Error: {exec_result.error}")


def main():
    models = sys.argv[1:] if len(sys.argv) > 1 else ["llama3.1:8b"]
    print(f"Comparing models: {models}")
    print("(Each model temporarily overrides config.MODEL for this run only.)")

    for model_name in models:
        run_for_model(model_name)

    print(f"\n{'=' * 70}")
    print("Done. Compare the outputs above by hand for now --")
    print("formal scoring across the full question set happens in eval/run_eval.py.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
