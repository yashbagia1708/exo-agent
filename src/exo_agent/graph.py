"""
The agent graph.

    question
      -> decomposer        split into sub-questions, route each to SQL and/or retrieval
      -> for each sub-question, the inner graph:
           sql_writer -> executor -> chart_decider -> [chart_generator]
           retriever                (in parallel with the SQL branch if both are needed)
           -> insight_writer        grounded sub-answer + caveat + sources
      -> synthesizer       paste the sub-answers together (no LLM call)

Sub-questions run one after another, not with LangGraph's Send fan-out:
most questions only have one, the trace stays in order, and one local
Ollama instance wouldn't run them any faster in parallel.

    python3 -m exo_agent.graph "How many planets were found by transit?"
"""

from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from exo_agent.nodes.chart_decider import decide_chart
from exo_agent.nodes.chart_generator import generate_chart
from exo_agent.nodes.entity_resolver import ResolvedEntity, resolve_entity
from exo_agent.nodes.executor import ExecutionResult, execute_sql
from exo_agent.nodes.insight_writer import write_insight
from exo_agent.nodes.planner import plan_decomposition
from exo_agent.nodes.sql_writer import write_sql
from exo_agent.retrieval import RetrievedChunk, hybrid_search
from exo_agent.schemas import (
    ChartSpec,
    DecompositionPlan,
    Insight,
    SQLQuery,
    SubQuestion,
)


class SubQuestionState(TypedDict, total=False):
    """One sub-question's run through the inner graph."""

    sub_question: SubQuestion
    resolved_entity: Optional[ResolvedEntity]
    sql_query: Optional[SQLQuery]
    execution_result: Optional[ExecutionResult]
    chart_spec: Optional[ChartSpec]
    chart_path: Optional[str]
    retrieved_chunks: Optional[list[RetrievedChunk]]
    retrieval_error: Optional[str]
    insight: Optional[Insight]
    error: Optional[str]


class AgentState(TypedDict, total=False):
    """The whole question: plan, each sub-question's result, final answer."""

    question: str
    decomposition: Optional[DecompositionPlan]
    sub_results: list[SubQuestionState]
    final_insight: Optional[Insight]
    error: Optional[str]


# ------------------------------------------------------------------ inner graph (one sub-question)


def sql_writer_node(state: SubQuestionState) -> SubQuestionState:
    question = state["sub_question"].text
    try:
        resolved_entity = resolve_entity(question)
    except Exception as e:
        print(f"  [sql_writer] Entity resolution failed ({e}), proceeding without it.")
        resolved_entity = None

    try:
        result = write_sql(question, resolved_entity=resolved_entity)
        return {"sql_query": result, "resolved_entity": resolved_entity}
    except Exception as e:
        return {"error": f"sql_writer failed: {e}", "resolved_entity": resolved_entity}


def executor_node(state: SubQuestionState) -> SubQuestionState:
    if state.get("error"):
        return {}
    sql_query = state["sql_query"]
    result = execute_sql(sql_query.sql, state["sub_question"].text)
    if not result.success:
        return {"execution_result": result, "error": f"executor failed: {result.error}"}
    return {"execution_result": result}


def chart_decider_node(state: SubQuestionState) -> SubQuestionState:
    if state.get("error"):
        return {}
    exec_result = state["execution_result"]
    spec = decide_chart(state["sub_question"].text, exec_result.columns, exec_result.rows)
    return {"chart_spec": spec}


def chart_generator_node(state: SubQuestionState) -> SubQuestionState:
    if state.get("error"):
        return {}
    spec = state["chart_spec"]
    exec_result = state["execution_result"]
    try:
        path = generate_chart(spec, exec_result.rows)
        return {"chart_path": str(path) if path else None}
    except ValueError as e:
        print(f"  [chart_generator] Skipping chart due to error: {e}")
        return {"chart_path": None}


def retriever_node(state: SubQuestionState) -> SubQuestionState:
    try:
        chunks = hybrid_search(state["sub_question"].text, top_k=5)
        return {"retrieved_chunks": chunks}
    except Exception as e:
        # Catch everything: the SQL half may still answer the question.
        # Log the exception type too; a bare "wiki_chunks does not exist"
        # once looked like "no results" and took ages to spot.
        print(
            f"  [retriever] RETRIEVAL FAILED ({type(e).__name__}): {e}\n"
            f"  [retriever] Continuing without retrieved passages for this sub-question."
        )
        return {"retrieved_chunks": [], "retrieval_error": f"{type(e).__name__}: {e}"}


def insight_writer_node(state: SubQuestionState) -> SubQuestionState:
    if state.get("error"):
        return {}
    exec_result = state.get("execution_result")
    rows = exec_result.rows if exec_result else []
    chart_generated = bool(state.get("chart_path"))
    chunks = state.get("retrieved_chunks") or []

    insight = write_insight(
        state["sub_question"].text, rows, chart_generated=chart_generated, retrieved_chunks=chunks
    )
    return {"insight": insight}


def route_sub_question_start(state: SubQuestionState) -> list[str]:
    """needs_sql / needs_retrieval -> which branch(es) to start. Neither -> SQL."""
    sub_q = state["sub_question"]
    branches = []
    if sub_q.needs_sql:
        branches.append("sql_writer")
    if sub_q.needs_retrieval:
        branches.append("retriever")
    return branches if branches else ["sql_writer"]


def route_after_chart_decision(state: SubQuestionState) -> str:
    if state.get("error"):
        return "insight_writer"
    spec = state.get("chart_spec")
    if spec and spec.chart_type != "none":
        return "chart_generator"
    return "insight_writer"


def build_sub_question_graph():
    """Compiled once at import, invoked per sub-question."""
    graph = StateGraph(SubQuestionState)

    graph.add_node("sql_writer", sql_writer_node)
    graph.add_node("executor", executor_node)
    graph.add_node("chart_decider", chart_decider_node)
    graph.add_node("chart_generator", chart_generator_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("insight_writer", insight_writer_node)

    graph.add_conditional_edges(
        START,
        route_sub_question_start,
        {"sql_writer": "sql_writer", "retriever": "retriever"},
    )

    graph.add_edge("sql_writer", "executor")
    graph.add_edge("executor", "chart_decider")
    graph.add_conditional_edges(
        "chart_decider",
        route_after_chart_decision,
        {"chart_generator": "chart_generator", "insight_writer": "insight_writer"},
    )
    graph.add_edge("chart_generator", "insight_writer")
    graph.add_edge("retriever", "insight_writer")
    graph.add_edge("insight_writer", END)

    return graph.compile()


_SUB_GRAPH = build_sub_question_graph()


# ------------------------------------------------------------------ outer graph (whole question)

def decomposer_node(state: AgentState) -> AgentState:
    try:
        plan = plan_decomposition(state["question"])
        return {"decomposition": plan}
    except Exception as e:
        print(f"  [decomposer] Decomposition failed ({e}), treating as a single SQL question.")
        fallback_sub = SubQuestion(text=state["question"], needs_sql=True, needs_retrieval=False)
        return {
            "decomposition": DecompositionPlan(
                sub_questions=[fallback_sub], reasoning="Fallback after decomposer error."
            )
        }


def run_sub_questions_node(state: AgentState) -> AgentState:
    """Run the inner graph for each sub-question, in order."""
    plan = state["decomposition"]
    results = []
    for sub_q in plan.sub_questions:
        print(f"  [sub-question] {sub_q.text}")
        sub_result = _SUB_GRAPH.invoke({"sub_question": sub_q})
        results.append(sub_result)
    return {"sub_results": results}


def synthesizer_node(state: AgentState) -> AgentState:
    """
    Paste the sub-answers together, word for word. No LLM call.

    This used to be an LLM rewrite, and it broke twice: it invented a
    "knowledge cutoff" line, then said there were no 2025 planets when
    the sub-answer had 200 rows (FINDINGS.md, 2026-06-23). Pasting can't
    add or contradict anything.
    """
    sub_results = state["sub_results"]

    # one sub-question (the usual case) -> its answer as-is
    if len(sub_results) == 1:
        single = sub_results[0]
        if single.get("error"):
            return {"error": single["error"]}
        return {"final_insight": single["insight"]}

    if not any(r.get("insight") and not r.get("error") for r in sub_results):
        return {"error": "All sub-questions failed; no answer could be synthesized."}

    paragraphs, caveats, sources = [], [], []
    for r in sub_results:
        insight = r.get("insight")
        if r.get("error") or not insight:
            # name the failed part instead of quietly leaving it out
            paragraphs.append(
                f'I couldn\'t answer "{r["sub_question"].text}": '
                f'{r.get("error") or "no answer was produced"}.'
            )
            continue
        paragraphs.append(insight.answer.strip())
        if insight.caveat:
            caveats.append(insight.caveat.strip())
        sources.extend(insight.sources_used)

    final = Insight(
        answer="\n\n".join(paragraphs),
        caveat=" ".join(dict.fromkeys(caveats)) or None,  # dedupe, keep order
        sources_used=list(dict.fromkeys(sources)),
    )
    return {"final_insight": final}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("decomposer", decomposer_node)
    graph.add_node("run_sub_questions", run_sub_questions_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("decomposer")
    graph.add_edge("decomposer", "run_sub_questions")
    graph.add_edge("run_sub_questions", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


def ask(question: str) -> AgentState:
    """Run one question through the whole graph and return the final state."""
    app = build_graph()
    final_state = app.invoke({"question": question})
    return final_state


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "How many confirmed exoplanets are there?"
    print(f"Question: {question}\n")

    result = ask(question)

    plan = result.get("decomposition")
    if plan:
        print(f"Decomposition: {len(plan.sub_questions)} sub-question(s) -- {plan.reasoning}")
        for sq in plan.sub_questions:
            print(f"  - {sq.text} (sql={sq.needs_sql}, retrieval={sq.needs_retrieval})")
        print()

    if result.get("error"):
        print(f"PIPELINE ERROR: {result['error']}")
    else:
        insight = result["final_insight"]

        for i, sub_result in enumerate(result.get("sub_results", [])):
            exec_result = sub_result.get("execution_result")
            chunks = sub_result.get("retrieved_chunks") or []
            if exec_result:
                print(f"[sub {i + 1}] SQL: {exec_result.sql_used}")
                print(f"[sub {i + 1}] Rows returned: {exec_result.row_count}")
            if chunks:
                print(f"[sub {i + 1}] Retrieved {len(chunks)} passage(s): "
                      f"{[c.title for c in chunks]}")
            if sub_result.get("chart_path"):
                print(f"[sub {i + 1}] Chart saved to: {sub_result['chart_path']}")

        print(f"\nFinal Answer: {insight.answer}")
        if insight.caveat:
            print(f"Caveat: {insight.caveat}")
        if insight.sources_used:
            print(f"Sources: {insight.sources_used}")
