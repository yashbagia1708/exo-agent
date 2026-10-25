"""Unit tests for merging dependent sub-questions. No model or database needed: `pytest -q`."""

import pytest

from exo_agent.nodes.planner import merge_dependent_parts
from exo_agent.schemas import DecompositionPlan, SubQuestion


def _plan(*parts, retrieval=()):
    subs = [SubQuestion(text=t, needs_sql=True, needs_retrieval=i in retrieval) for i, t in enumerate(parts)]
    return DecompositionPlan(sub_questions=subs, reasoning="Two asks.")


@pytest.mark.parametrize("parts, merged", [
    # what the model actually returned, 3/3 runs each
    (["Which planet has the most Earth-like conditions?", "How far is this planet from Earth?"],
     "Which planet has the most Earth-like conditions, and how far is this planet from Earth?"),
    (["Which star has the most planets?", "What spectral type is this star?"],
     "Which star has the most planets, and what spectral type is this star?"),
    (["What is the largest planet found by TESS?", "When was it discovered?"],
     "What is the largest planet found by TESS, and when was it discovered?"),
    (["Which exoplanet is the largest found by TESS?", "When was this exoplanet discovered?"],
     "Which exoplanet is the largest found by TESS, and when was this exoplanet discovered?"),
    (["What is the largest planet found by TESS?", "When was this largest planet discovered?"],
     "What is the largest planet found by TESS, and when was this largest planet discovered?"),
])
def test_back_references_are_folded_into_the_part_before(parts, merged):
    out = merge_dependent_parts(_plan(*parts))
    assert [s.text for s in out.sub_questions] == [merged]
    assert "refers back" in out.reasoning


@pytest.mark.parametrize("parts", [
    ["How many planets were found by the transit method?", "What is the habitable zone?"],
    ["List the planets orbiting TRAPPIST-1", "What is the Kepler space telescope?"],
    ["How many planets were found this year?", "How many planets orbit M dwarfs that are nearby?"],
])
def test_independent_parts_stay_split(parts):
    plan = _plan(*parts)
    assert merge_dependent_parts(plan) is plan


def test_merged_part_keeps_both_tools():
    out = merge_dependent_parts(_plan("How many planets did Kepler find?", "How did it find them?", retrieval={1}))
    (only,) = out.sub_questions
    assert only.needs_sql and only.needs_retrieval
