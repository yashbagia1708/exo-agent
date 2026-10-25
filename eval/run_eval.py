"""
Eval: run every question in eval/questions.jsonl N times (default 5) and
report a pass rate per question, e.g. "4/5".

One clean run proves little, because the model isn't deterministic. The
checks are simple on purpose: does the SQL contain X, is the caveat set,
does the answer decline. No LLM judge.

    python3 eval/run_eval.py
    python3 eval/run_eval.py --runs 3              # quicker, rougher
    python3 eval/run_eval.py --tier 1              # one tier
    python3 eval/run_eval.py --id t1_total_count   # one question
"""

import argparse
import json
import pathlib
import sys
import time
from dataclasses import dataclass, field

# lets `python3 eval/run_eval.py` work without `pip install -e .`
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
    """sql_writer (+ executor if asked) -> string checks on the SQL."""
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
        # SQL can look right and still fail in Postgres (wrong types), so run it
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
    Full pipeline (sql_writer -> executor -> insight_writer), then check
    the final answer. For tier 5 ("which planet has confirmed life?"),
    where the SQL can run fine and the only thing to judge is the answer.

        should_decline_or_caveat  caveat set, or a hedge phrase in the answer
        should_decline            the answer itself must say there's no data;
                                  a caveat on a confident answer isn't enough
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
    """write_insight() on the spec's fake rows -> check the answer/caveat."""
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

    stage = spec.get("stage", "sql")

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
