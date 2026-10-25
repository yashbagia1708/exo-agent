"""
The planner node: the agent's first real decision point. Given a
question, decides (1) whether it's actually multiple distinct questions
joined together, and (2) for each resulting sub-question, whether it
needs structured SQL, conceptual retrieval (hybrid search over
Wikipedia), or both.

A message joining two unrelated asks with "and" (e.g. "list X, and tell
me Y") is two independent questions, not one question with two
conditions -- fusing them into a single SQL query would silently answer
neither part correctly. Most real questions still decompose to exactly
one sub-question; the multi-sub-question path exists specifically for
genuinely compound input.
"""

from exo_agent import config
from exo_agent.providers.factory import get_client
from exo_agent.schemas import DecompositionPlan

SYSTEM_PROMPT = """You analyze a question about exoplanets and decide two \
things: how many genuinely distinct questions are actually being asked, \
and which tool(s) each one needs.

Two tools are available:
1. SQL over a structured database of 6,298 confirmed exoplanets and their \
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
"""

USER_PROMPT = """Question: "{question}"

Decide how many distinct sub-questions this actually contains, and which \
tool(s) each needs.
"""


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
    return result


if __name__ == "__main__":
    # Pass a question as a CLI argument to test it directly, e.g.:
    #   python3 -m exo_agent.nodes.planner "your question here"
    # With no argument, runs the built-in regression questions instead --
    # covering single questions, conceptual questions, hybrid (both
    # tools) questions, and the real compound question that motivated
    # this node's decomposition behavior.
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
