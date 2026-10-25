"""
Planner: split the question into sub-questions and route each one.

    "list X, and tell me Y"  -> 2 sub-questions (one SQL query can't answer both)
    "planets > Earth after 2015" -> 1 sub-question with two filters

Each sub-question gets needs_sql and/or needs_retrieval (Wikipedia search).
"""

import re

from exo_agent import config
from exo_agent.providers.factory import get_client
from exo_agent.schemas import DecompositionPlan, SubQuestion

# a later part saying "it", "its" or "this (largest) planet" depends on an earlier one.
# "this/that" only count before a noun like planet/star (up to 2 words between),
# so "this year" or "planets that orbit..." don't trigger it
BACK_REFERENCE = re.compile(
    r"\b(it|its|they|them|their)\b"
    r"|\b(this|that|these|those|the same) (\w+ ){0,2}(planet|exoplanet|star|system|world|one|object)s?\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You analyze a question about exoplanets and decide two \
things: how many genuinely distinct questions are actually being asked, \
and which tool(s) each one needs.

Two tools are available:
1. SQL over a structured database of 6,000+ confirmed exoplanets and their \
host stars (counts, filters, comparisons, trends, specific named planets/ \
stars -- anything with a real numeric or categorical answer in the data).
2. Hybrid search (retrieval) over a small corpus of Wikipedia articles \
covering detection methods, astronomy concepts, and a few notable systems \
(answers "what is X", "how does X work", "why does X happen" -- conceptual \
questions with no column in the database).

Most input is exactly ONE sub-question -- only split into more when the \
message genuinely joins unrelated asks (e.g. "give me X, and also tell me \
Y" where X and Y are about different things). Do NOT split a single \
question that simply has multiple filter conditions -- "planets bigger \
than Earth discovered after 2015" is one sub-question with two filters, \
not two sub-questions. If a sub-question could plausibly be answered by \
SQL alone, prefer needs_sql=true, needs_retrieval=false even if the topic \
sounds conceptual at a glance.

Never split off a part that depends on another part's answer ("which \
planet is X, and how far is it?" is ONE sub-question).

Keep the user's meaning when you rewrite a sub-question. "How far is it" \
with no reference point means distance from Earth, not from its host star.
"""

USER_PROMPT = """Question: "{question}"

Decide how many distinct sub-questions this actually contains, and which \
tool(s) each needs.
"""


def merge_dependent_parts(plan: DecompositionPlan) -> DecompositionPlan:
    """
    Fold any part that points back at an earlier one into it.

        ["Which star has the most planets?", "What spectral type is this star?"]
        -> ["Which star has the most planets, and what spectral type is this star?"]

    Sub-questions run separately, so "this star" on its own has nothing to
    refer to; it used to guess (once it searched for Proxima). The model
    split these 9/9 times even with a prompt rule against it, so it's
    enforced here.
    """
    merged: list[SubQuestion] = []
    folded = []
    for i, sq in enumerate(plan.sub_questions):
        if merged and BACK_REFERENCE.search(sq.text):
            prev = merged[-1]
            text = f"{prev.text.rstrip(' ?.')}, and {sq.text[0].lower()}{sq.text[1:]}"
            merged[-1] = SubQuestion(
                text=text,
                needs_sql=prev.needs_sql or sq.needs_sql,
                needs_retrieval=prev.needs_retrieval or sq.needs_retrieval,
            )
            folded.append(i + 1)
        else:
            merged.append(sq)

    if not folded:
        return plan
    note = f" (Part {', '.join(map(str, folded))} refers back to an earlier part, so they're answered together.)"
    return DecompositionPlan(sub_questions=merged, reasoning=plan.reasoning + note)


def plan_decomposition(question: str) -> DecompositionPlan:
    client = get_client()

    result = client.chat.completions.create(
        model=config.MODEL,
        response_model=DecompositionPlan,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(question=question)},
        ],
    )
    return merge_dependent_parts(result)


if __name__ == "__main__":
    # python3 -m exo_agent.nodes.planner "question"   -> plan one question
    # python3 -m exo_agent.nodes.planner              -> run the sample set below
    # (single, conceptual, both tools, and the compound question that started this)
    import sys

    if len(sys.argv) > 1:
        test_questions = [sys.argv[1]]
    else:
        test_questions = [
            "How many confirmed exoplanets are there?",
            "How does the transit method actually work?",
            "How many planets were found by the transit method, and how does that method actually work?",
            "Give me a list of exoplanets that are closer to their host star than Earth and tell me the equilibrium temperature of Proxima Centauri b",
            "Which planets bigger than Earth were discovered after 2015?",
        ]

    for q in test_questions:
        print(f"\nQuestion: {q}")
        plan = plan_decomposition(q)
        print(f"  reasoning: {plan.reasoning}")
        for i, sub in enumerate(plan.sub_questions):
            print(f"  [{i + 1}] {sub.text}")
            print(f"      needs_sql={sub.needs_sql}, needs_retrieval={sub.needs_retrieval}")
