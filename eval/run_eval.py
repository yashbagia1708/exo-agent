"""
The eval harness: runs every question in eval/questions.jsonl multiple
times each (default 5) and reports a pass rate per question, not a
single pass/fail.

LLM behavior on identical input is not deterministic. A single clean run
of a question can be misleading -- a prompt fix can genuinely improve a
model's odds of getting something right without making it reliable every
time. Running each question 5x and reporting, for example, "4/5 passed"
is the honest way to represent that.

This harness is deliberately simple string/structural checking, not deep
semantic scoring -- it checks things like "does the SQL contain this
substring" or "is the caveat field non-null", which is enough to catch
known regression patterns without building a much heavier LLM-as-judge
scoring system. That's a reasonable next iteration, not a requirement for
this to be useful now.

Usage:
    PROVIDER=ollama python3 eval/run_eval.py
    PROVIDER=ollama python3 eval/run_eval.py --runs 3          # faster, rougher
    PROVIDER=ollama python3 eval/run_eval.py --tier 1           # only tier 1 questions
    PROVIDER=ollama python3 eval/run_eval.py --id t1_total_count  # just one question
"""

import argparse
import json
import pathlib
import sys
import time
from dataclasses import dataclass, field

# Allow running this script directly (python3 eval/run_eval.py) without
# having installed the package -- add src/ to the path if needed.
_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from exo_agent import config
from exo_agent.nodes.executor import execute_sql
from exo_agent.nodes.insight_writer import write_insight
from exo_agent.nodes.sql_writer import write_sql

QUESTIONS_PATH = pathlib.Path(__file__).parent / "questions.jsonl"


@dataclass
class RunResult:
    passed: bool
    detail: str
    elapsed_seconds: float
    repair_attempts: int = 0


@dataclass
class QuestionReport:
    question_id: str
    tier: int
    question_text: str
    runs: list[RunResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> str:
        passed = sum(1 for r in self.runs if r.passed)
        return f"{passed}/{len(self.runs)}"

    @property
    def avg_elapsed(self) -> float:
        if not self.runs:
            return 0.0
        return sum(r.elapsed_seconds for r in self.runs) / len(self.runs)


def _check_sql_stage(spec: dict) -> RunResult:
    """Run sql_writer (+ optionally executor) and check string-level expectations."""
    start = time.time()
    try:
        sql_result = write_sql(spec["question"])
    except Exception as e:
        return RunResult(False, f"write_sql raised: {e}", time.time() - start)

    sql_upper = sql_result.sql.upper()
    failures = []

    for expected in spec.get("expected_sql_contains", []):
        if expected.upper() not in sql_upper:
            failures.append(f"missing expected substring '{expected}'")

    for excluded in spec.get("expected_sql_excludes", []):
        if excluded.upper() in sql_upper:
            failures.append(f"contains excluded substring '{excluded}'")

    if spec.get("expect_chart") is True and not sql_result.needs_chart:
        failures.append("expected needs_chart=True but got False")

    repair_attempts = 0
    if not failures:
        # Also verify it actually executes -- structural checks alone
        # don't catch semantic errors, e.g. a type mismatch that's valid
        # SQL syntax but fails at runtime against the real schema.
        exec_result = execute_sql(sql_result.sql, spec["question"])
        repair_attempts = exec_result.repair_attempts
        if not exec_result.success:
            failures.append(f"execution failed: {exec_result.error}")

    elapsed = time.time() - start
    if failures:
        return RunResult(False, "; ".join(failures), elapsed, repair_attempts)
    return RunResult(True, f"sql={sql_result.sql}", elapsed, repair_attempts)


def _check_behavior_stage(spec: dict) -> RunResult:
    """
    Run the FULL pipeline (sql_writer -> executor -> insight_writer) and
    check whether the final answer appropriately declines or caveats,
    per spec["expected_behavior"].

    This exists specifically for tier-5 questions like "which planet has
    confirmed life on it" -- there's no SQL-level string check that can
    capture "did the agent appropriately refuse to overclaim", since the
    SQL itself might execute fine (e.g. a query that just returns NULL
    columns or an empty/meaningless result). The only place this can
    actually be judged is the final Insight object, after real data has
    come back from the database.

    Two behaviors are supported:
      - "should_decline_or_caveat": caveat must be non-null, OR the answer
        itself must contain a hedge phrase (covers both Insight design
        paths -- a model might fold the caveat into the main answer text
        instead of using the caveat field, which isn't wrong, just a
        different valid shape).
      - "should_decline": the answer must contain an explicit refusal/
        no-data phrase, not just any caveat -- this is for genuinely
        unanswerable questions (e.g. "which planet has life"), where a
        caveated-but-still-confident answer isn't good enough.
    """
    start = time.time()
    HEDGE_PHRASES = [
        "not available", "no data", "cannot", "can't", "not measured",
        "not directly", "doesn't measure", "does not measure", "proxy",
        "no column", "not in the dataset", "this data", "the dataset",
        "no direct", "isn't measured", "is not measured",
    ]
    DECLINE_PHRASES = [
        "cannot determine", "can't determine", "no data", "not possible to",
        "no way to", "not something", "doesn't tell us", "does not tell us",
        "no column", "not available in", "cannot answer", "can't answer",
    ]

    try:
        sql_result = write_sql(spec["question"])
        exec_result = execute_sql(sql_result.sql, spec["question"])
        if not exec_result.success:
            elapsed = time.time() - start
            return RunResult(
                False, f"pipeline failed before insight stage: {exec_result.error}", elapsed
            )
        insight = write_insight(
            spec["question"], exec_result.rows, chart_generated=False
        )
    except Exception as e:
        return RunResult(False, f"pipeline raised: {e}", time.time() - start)

    elapsed = time.time() - start
    combined_text = f"{insight.answer} {insight.caveat or ''}".lower()

    behavior = spec.get("expected_behavior")
    if behavior == "should_decline_or_caveat":
        ok = bool(insight.caveat) or any(p in combined_text for p in HEDGE_PHRASES)
        detail = f"answer={insight.answer!r} caveat={insight.caveat!r}"
        if not ok:
            return RunResult(False, f"no caveat or hedge phrase found | {detail}", elapsed)
        return RunResult(True, detail, elapsed)

    if behavior == "should_decline":
        ok = any(p in combined_text for p in DECLINE_PHRASES)
        detail = f"answer={insight.answer!r} caveat={insight.caveat!r}"
        if not ok:
            return RunResult(False, f"no decline phrase found | {detail}", elapsed)
        return RunResult(True, detail, elapsed)

    return RunResult(False, f"unknown expected_behavior: {behavior!r}", elapsed)


def _check_insight_stage(spec: dict) -> RunResult:
    """Run write_insight() directly against synthetic rows and check expectations."""
    start = time.time()
    try:
        result = write_insight(spec["question"], spec["synthetic_rows"])
    except Exception as e:
        return RunResult(False, f"write_insight raised: {e}", time.time() - start)

    failures = []

    if spec.get("expected_caveat_not_null") and not result.caveat:
        failures.append("expected a non-null caveat but got None")

    answer_lower = result.answer.lower()
    for excluded in spec.get("expected_answer_excludes", []):
        if excluded.lower() in answer_lower:
            failures.append(f"answer contains excluded text '{excluded}'")

    elapsed = time.time() - start
    if failures:
        return RunResult(False, "; ".join(failures) + f" | answer={result.answer!r}", elapsed)
    return RunResult(True, f"answer={result.answer!r} caveat={result.caveat!r}", elapsed)


def run_question(spec: dict, num_runs: int) -> QuestionReport:
    report = QuestionReport(
        question_id=spec["id"], tier=spec["tier"], question_text=spec["question"]
    )

    stage = spec.get("stage", "sql")  # default: full SQL pipeline check

    for i in range(num_runs):
        if stage == "insight_writer":
            result = _check_insight_stage(spec)
        elif spec.get("expected_behavior"):
            result = _check_behavior_stage(spec)
        else:
            result = _check_sql_stage(spec)
        report.runs.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"    run {i + 1}/{num_runs}: {status} ({result.elapsed_seconds:.1f}s)")
        if not result.passed:
            print(f"      {result.detail}")

    return report


def load_questions(tier_filter: int | None = None, id_filter: str | None = None) -> list[dict]:
    specs = []
    with open(QUESTIONS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            spec = json.loads(line)
            if tier_filter is not None and spec["tier"] != tier_filter:
                continue
            if id_filter is not None and spec["id"] != id_filter:
                continue
            specs.append(spec)
    return specs


def main():
    parser = argparse.ArgumentParser(description="Run the exo-agent eval suite.")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per question (default: 5)")
    parser.add_argument("--tier", type=int, default=None, help="Only run questions in this tier")
    parser.add_argument("--id", type=str, default=None, help="Only run the question with this id")
    args = parser.parse_args()

    specs = load_questions(tier_filter=args.tier, id_filter=args.id)
    if not specs:
        print("No questions matched the given filters.")
        return

    print(f"Running {len(specs)} question(s), {args.runs} run(s) each, against {config.describe()}\n")

    reports = []
    overall_start = time.time()

    for spec in specs:
        print(f"[{spec['id']}] (tier {spec['tier']}) {spec['question']}")
        report = run_question(spec, args.runs)
        reports.append(report)
        print(f"  -> pass rate: {report.pass_rate}\n")

    overall_elapsed = time.time() - overall_start

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'ID':<30} {'Tier':<6} {'Pass rate':<10} {'Avg time':<10}")
    for r in reports:
        print(f"{r.question_id:<30} {r.tier:<6} {r.pass_rate:<10} {r.avg_elapsed:.1f}s")

    total_runs = sum(len(r.runs) for r in reports)
    total_passed = sum(sum(1 for run in r.runs if run.passed) for r in reports)
    print(f"\nOverall: {total_passed}/{total_runs} individual runs passed "
          f"({100 * total_passed / total_runs:.0f}%)")
    print(f"Total eval time: {overall_elapsed:.1f}s")
    print(f"Config: {config.describe()}")


if __name__ == "__main__":
    main()
