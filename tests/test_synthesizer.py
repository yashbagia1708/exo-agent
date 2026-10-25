"""
Regression tests for synthesizer_node, replaying the 2026-06-23 bugs:
an invented "knowledge cutoff" line, and "no 2025 planets" over 200 real rows.
No model or database needed: `pytest -q`.
"""

import pytest

from exo_agent import graph
from exo_agent.schemas import Insight, SubQuestion

SUB_2025 = SubQuestion(
    text="Give me a list of exoplanets discovered in 2025", needs_sql=True, needs_retrieval=False
)
SUB_RV = SubQuestion(
    text="Which exoplanet has the highest radial velocity?", needs_sql=True, needs_retrieval=False
)
SUB_KEPLER = SubQuestion(
    text="What is the Kepler space telescope?", needs_sql=False, needs_retrieval=True
)

ANSWER_2025 = (
    "200 exoplanets in the dataset were discovered in 2025, including "
    "TOI-6324 b, TOI-5795 b and TOI-2431 b."
)
ANSWER_RV = (
    "The dataset has no measured radial velocity values, so no planet can be "
    "ranked by it. It only records whether a planet was discovered using the "
    "radial velocity method."
)
CAVEAT_RV = "Radial velocity is only recorded as a discovery method, not as a measured value."
KEPLER_URL = "https://en.wikipedia.org/wiki/Kepler_space_telescope"


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """Synthesis must never call the model again; make any call blow up."""

    def _boom(*args, **kwargs):
        raise AssertionError("synthesizer_node must not call the LLM")

    monkeypatch.setattr("exo_agent.providers.factory.get_client", _boom)
    monkeypatch.setattr(graph, "get_client", _boom, raising=False)


def _sub(sub_q, answer=None, caveat=None, sources=None, error=None):
    result = {"sub_question": sub_q}
    if answer is not None:
        result["insight"] = Insight(answer=answer, caveat=caveat, sources_used=sources or [])
    if error:
        result["error"] = error
    return result


def _synthesize(question, *sub_results):
    return graph.synthesizer_node({"question": question, "sub_results": list(sub_results)})


def test_compound_repro_keeps_populated_answer_and_adds_nothing():
    # the 2026-06-23 repro: one good sub-answer, one unanswerable
    out = _synthesize(
        "Give me a list of exoplanets discovered in 2025. In general, which "
        "exoplanet has the highest radial velocity?",
        _sub(SUB_2025, ANSWER_2025),
        _sub(SUB_RV, ANSWER_RV, caveat=CAVEAT_RV),
    )
    answer = out["final_insight"].answer

    assert ANSWER_2025 in answer and ANSWER_RV in answer
    assert answer.index(ANSWER_2025) < answer.index(ANSWER_RV)
    for forbidden in ("knowledge cutoff", "as an ai", "training data", "no exoplanets"):
        assert forbidden not in answer.lower()
    # exactly the two answers and a blank line, nothing added
    assert answer == f"{ANSWER_2025}\n\n{ANSWER_RV}"


def test_caveats_and_sources_carried_over_and_deduplicated():
    out = _synthesize(
        "compound",
        _sub(SUB_RV, ANSWER_RV, caveat=CAVEAT_RV),
        _sub(SUB_KEPLER, "Kepler was a NASA space telescope.", caveat=CAVEAT_RV, sources=[KEPLER_URL]),
        _sub(SUB_KEPLER, "It launched in 2009.", sources=[KEPLER_URL]),
    )
    final = out["final_insight"]

    assert final.caveat == CAVEAT_RV
    assert final.sources_used == [KEPLER_URL]


def test_failed_part_is_reported_not_dropped_or_used_to_hedge():
    out = _synthesize(
        "compound",
        _sub(SUB_2025, ANSWER_2025),
        _sub(SUB_RV, error="executor failed: column \"radial_velocity\" does not exist"),
    )
    answer = out["final_insight"].answer

    assert answer.startswith(ANSWER_2025)
    assert f'couldn\'t answer "{SUB_RV.text}"' in answer
    assert "radial_velocity" in answer


def test_single_sub_question_passes_through_and_all_failed_is_an_error():
    single = _sub(SUB_2025, ANSWER_2025, caveat="Some caveat.")
    assert _synthesize("q", single)["final_insight"] is single["insight"]

    assert _synthesize("q", _sub(SUB_2025, error="sql_writer failed: boom")) == {
        "error": "sql_writer failed: boom"
    }
    assert "error" in _synthesize("q", _sub(SUB_2025, error="a"), _sub(SUB_RV, error="b"))
