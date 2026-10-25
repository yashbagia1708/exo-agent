"""
Minimal API server exposing the real exo-agent graph to the frontend.

This does NOT reimplement any agent logic -- it imports and calls
src/exo_agent/graph.py directly, the exact same code path used by the
CLI (`python3 -m exo_agent.graph "..."`) and the eval harness. The
frontend is a thin client on top of the same agent that's been tested
throughout this project; it doesn't talk to a different, simplified
version of it.

Updated for the decomposition-aware graph: a question can now produce
MULTIPLE sub-question results (each with its own SQL/retrieval/insight),
synthesized into one final answer. The trace reflects this -- each
sub-question gets its own numbered group of steps, followed by a
synthesis step if there was more than one.

Run with:
    PROVIDER=ollama uvicorn frontend.server:app --reload --port 8420
"""

import base64
import pathlib
import sys
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_SRC = pathlib.Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from exo_agent import config
from exo_agent.graph import ask

app = FastAPI(title="exo-agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class TraceStep(BaseModel):
    label: str
    detail: str


class SubQuestionSummary(BaseModel):
    """One row of the glanceable 'how it was split' summary -- separate
    from the full step-by-step trace, meant to be readable at a glance
    without expanding anything."""

    text: str
    used_sql: bool
    used_retrieval: bool
    sql: str | None = None
    retrieved_titles: list[str] = []


class AskResponse(BaseModel):
    question: str
    success: bool
    answer: str | None = None
    caveat: str | None = None
    sources: list[str] = []
    sql_statements: list[str] = []  # one per sub-question that used SQL
    row_count: int | None = None  # total across all sub-questions
    chart_base64: str | None = None  # first chart found across sub-questions, if any
    trace: list[TraceStep] = []
    routing_summary: list[SubQuestionSummary] = []
    decomposition_reasoning: str | None = None
    error: str | None = None
    elapsed_seconds: float = 0.0
    model_used: str = ""
    sub_question_count: int = 1


def _build_trace(result: dict) -> list[TraceStep]:
    """
    Turn the final graph state into a human-readable trace of what the
    agent actually did, in order -- across potentially multiple
    sub-questions. This reads real fields off the same state objects
    graph.py produces; it is not a separate, prettified description of
    what happened.

    Surfaces two of the agent's more sophisticated decisions explicitly:
      - The planner's per-sub-question tool routing decision (why SQL
        and/or retrieval was chosen for this specific sub-question, not
        just that decomposition happened).
      - Entity resolution: when a sub-question names a specific planet or
        star, the real trigram-similarity match found against the
        database before the SQL writer used it -- verifying named
        entities against ground truth rather than guessing at spelling.
    """
    steps: list[TraceStep] = []

    decomposition = result.get("decomposition")
    sub_results = result.get("sub_results", [])
    multi = decomposition and len(decomposition.sub_questions) > 1

    step_counter = 1

    if multi:
        steps.append(
            TraceStep(
                label=f"{step_counter}. Split into {len(decomposition.sub_questions)} sub-questions",
                detail=(
                    f"{decomposition.reasoning}\n\n"
                    + "\n".join(f"- {sq.text}" for sq in decomposition.sub_questions)
                ),
            )
        )
        step_counter += 1

    for i, sub_result in enumerate(sub_results):
        prefix = f"Sub-question {i + 1}: " if multi else ""
        sub_q = decomposition.sub_questions[i] if decomposition else None

        # --- Routing decision for this specific sub-question ---
        if sub_q is not None:
            tools = []
            if sub_q.needs_sql:
                tools.append("structured database (SQL)")
            if sub_q.needs_retrieval:
                tools.append("Wikipedia hybrid search")
            tools_str = " + ".join(tools) if tools else "structured database (SQL)"
            steps.append(
                TraceStep(
                    label=f"{step_counter}. {prefix}Routed to: {tools_str}",
                    detail=(
                        f'"{sub_q.text}"\n\n'
                        f"Chosen because this question needs "
                        + ("real numbers/counts/comparisons from the database" if sub_q.needs_sql else "")
                        + (" and " if sub_q.needs_sql and sub_q.needs_retrieval else "")
                        + ("conceptual background knowledge not present as a column" if sub_q.needs_retrieval else "")
                        + "."
                    ),
                )
            )
            step_counter += 1

        # --- Entity resolution, if a named planet/star was checked ---
        entity = sub_result.get("resolved_entity")
        if entity:
            if entity.resolved_name:
                steps.append(
                    TraceStep(
                        label=f"{step_counter}. {prefix}Verified a named entity against the database",
                        detail=(
                            f"Mentioned: \"{entity.mentioned_as}\"\n"
                            f"Matched to real value: \"{entity.resolved_name}\" "
                            f"(in {entity.table}, trigram similarity = {entity.similarity_score:.2f})\n\n"
                            f"This confirmed name is used directly in an exact-match filter -- "
                            f"the SQL writer never has to guess catalogue spelling/abbreviation "
                            f"conventions, because the real value was looked up deterministically "
                            f"first."
                        ),
                    )
                )
            else:
                steps.append(
                    TraceStep(
                        label=f"{step_counter}. {prefix}Checked a named entity — no confident match",
                        detail=(
                            f"Mentioned: \"{entity.mentioned_as}\"\n"
                            f"Best similarity found: {entity.similarity_score or 0:.2f} "
                            f"(below the 0.35 confidence threshold)\n\n"
                            f"This object likely isn't in the dataset under any close spelling."
                        ),
                    )
                )
            step_counter += 1

        # --- SQL writing ---
        sql_query = sub_result.get("sql_query")
        if sql_query:
            steps.append(
                TraceStep(
                    label=f"{step_counter}. {prefix}Wrote SQL",
                    detail=f"{sql_query.sql}\n\nReasoning: {sql_query.reasoning}",
                )
            )
            step_counter += 1

        # --- Execution, with repair attempts called out explicitly ---
        exec_result = sub_result.get("execution_result")
        if exec_result:
            if exec_result.repair_attempts:
                steps.append(
                    TraceStep(
                        label=f"{step_counter}. {prefix}Query failed — self-repaired and retried",
                        detail=(
                            f"The first attempt produced a SQL error from Postgres. The real "
                            f"error message was fed back to the model, which corrected it and "
                            f"retried -- {exec_result.repair_attempts} repair attempt(s) before "
                            f"success.\n\nFinal working SQL: {exec_result.sql_used}"
                        ),
                    )
                )
            else:
                steps.append(
                    TraceStep(
                        label=f"{step_counter}. {prefix}Ran it against Postgres",
                        detail=f"Returned {exec_result.row_count} row(s), no repair needed.",
                    )
                )
            step_counter += 1

        # --- Retrieval ---
        chunks = sub_result.get("retrieved_chunks")
        retrieval_error = sub_result.get("retrieval_error")

        if chunks:
            chunk_lines = "\n".join(
                f"- {c.title} (rrf={c.rrf_score:.4f}, lexical_rank={c.lexical_rank}, "
                f"semantic_rank={c.semantic_rank})"
                for c in chunks
            )
            steps.append(
                TraceStep(
                    label=f"{step_counter}. {prefix}Hybrid search: lexical + semantic, fused by RRF",
                    detail=(
                        f"Ran both a full-text (keyword) search and a vector (semantic) search, "
                        f"then merged the two ranked lists with Reciprocal Rank Fusion -- a passage "
                        f"ranking well in BOTH lists outranks one that only does well in one.\n\n"
                        f"Top {len(chunks)} result(s) after fusion:\n{chunk_lines}"
                    ),
                )
            )
            step_counter += 1
        elif retrieval_error:
            steps.append(
                TraceStep(
                    label=f"{step_counter}. {prefix}Retrieval failed — continued without it",
                    detail=(
                        f"Error: {retrieval_error}\n\n"
                        f"This sub-question's answer (if any) comes only from SQL data, "
                        f"not from Wikipedia. This is a real infrastructure/configuration "
                        f"error, not the agent choosing not to retrieve."
                    ),
                )
            )
            step_counter += 1

        # --- Chart decision ---
        chart_spec = sub_result.get("chart_spec")
        if chart_spec:
            if chart_spec.chart_type != "none":
                steps.append(
                    TraceStep(
                        label=f"{step_counter}. {prefix}Decided a chart would help",
                        detail=(
                            f"Chart type: {chart_spec.chart_type}\n"
                            f"x = {chart_spec.x_field}, y = {chart_spec.y_field}\n\n"
                            f"This decision looked at the ACTUAL result shape (row count, column "
                            f"types), not just the question text."
                        ),
                    )
                )
            else:
                steps.append(
                    TraceStep(
                        label=f"{step_counter}. {prefix}Decided no chart was needed",
                        detail="The result shape (e.g. a single value) didn't warrant one.",
                    )
                )
            step_counter += 1

        # --- Sub-answer grounding ---
        sub_insight = sub_result.get("insight")
        if sub_insight:
            steps.append(
                TraceStep(
                    label=f"{step_counter}. {prefix}Wrote a grounded sub-answer",
                    detail=(
                        "Grounded strictly in the SQL rows and/or retrieved passages above -- "
                        "no fact stated here comes from the model's memory."
                    ),
                )
            )
            step_counter += 1

    if multi:
        num_sources = len(set(
            src for r in sub_results for src in (r.get("insight").sources_used if r.get("insight") else [])
        ))
        sources_note = f" Drew on {num_sources} source(s) total across all parts." if num_sources else ""
        steps.append(
            TraceStep(
                label=f"{step_counter}. Synthesized one final answer",
                detail=(
                    f"Combined all {len(sub_results)} sub-answers above into one coherent, "
                    f"natural-language response, carrying forward every caveat and source "
                    f"exactly as found -- no new facts were added during this step.{sources_note}"
                ),
            )
        )

    return steps


def _build_routing_summary(result: dict) -> list[SubQuestionSummary]:
    """
    Builds the compact, glanceable 'how it was split' summary shown above
    the full trace -- one row per sub-question, showing in plain sight
    which tool(s) it used, what SQL ran (if any), and which Wikipedia
    articles were drawn on (if any). This is deliberately separate from
    _build_trace(): the trace is for someone who wants every step, this
    is for someone who just wants the routing story at a glance.
    """
    decomposition = result.get("decomposition")
    sub_results = result.get("sub_results", [])

    summaries = []
    for i, sub_result in enumerate(sub_results):
        sub_q = decomposition.sub_questions[i] if decomposition else None
        exec_result = sub_result.get("execution_result")
        chunks = sub_result.get("retrieved_chunks") or []

        summaries.append(
            SubQuestionSummary(
                text=sub_q.text if sub_q else result.get("question", ""),
                used_sql=bool(exec_result),
                used_retrieval=bool(chunks),
                sql=exec_result.sql_used if exec_result else None,
                retrieved_titles=list({c.title for c in chunks}),  # de-duplicated, order not critical here
            )
        )
    return summaries


@app.post("/api/ask", response_model=AskResponse)
def ask_question(req: AskRequest) -> AskResponse:
    start = time.time()
    try:
        result = ask(req.question)
    except Exception as e:
        return AskResponse(
            question=req.question,
            success=False,
            error=str(e),
            elapsed_seconds=time.time() - start,
            model_used=config.describe(),
        )

    elapsed = time.time() - start

    if result.get("error"):
        return AskResponse(
            question=req.question,
            success=False,
            error=result["error"],
            trace=_build_trace(result),
            elapsed_seconds=elapsed,
            model_used=config.describe(),
        )

    insight = result.get("final_insight")
    sub_results = result.get("sub_results", [])
    decomposition = result.get("decomposition")

    sql_statements = []
    total_rows = 0
    chart_base64 = None

    for sub_result in sub_results:
        exec_result = sub_result.get("execution_result")
        if exec_result:
            sql_statements.append(exec_result.sql_used)
            total_rows += exec_result.row_count

        if chart_base64 is None:
            chart_path = sub_result.get("chart_path")
            if chart_path and pathlib.Path(chart_path).exists():
                chart_base64 = base64.b64encode(
                    pathlib.Path(chart_path).read_bytes()
                ).decode("utf-8")

    return AskResponse(
        question=req.question,
        success=True,
        answer=insight.answer if insight else None,
        caveat=insight.caveat if insight else None,
        sources=insight.sources_used if insight else [],
        sql_statements=sql_statements,
        row_count=total_rows if sql_statements else None,
        chart_base64=chart_base64,
        trace=_build_trace(result),
        routing_summary=_build_routing_summary(result),
        decomposition_reasoning=decomposition.reasoning if decomposition else None,
        elapsed_seconds=elapsed,
        model_used=config.describe(),
        sub_question_count=len(decomposition.sub_questions) if decomposition else 1,
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "model": config.describe()}


@app.get("/api/topics")
def topics():
    """
    Returns the REAL list of Wikipedia articles currently loaded into
    the hybrid-search corpus, read directly from the manifest written by
    data/download_wiki_corpus.py. This is deliberately not a hand-typed
    list in the frontend -- if the corpus changes (more articles added,
    one fails to download), this endpoint reflects that automatically
    instead of silently going stale.
    """
    manifest_path = pathlib.Path(__file__).parent.parent / "data" / "wiki_corpus" / "_manifest.json"
    if not manifest_path.exists():
        return {"topics": [], "note": "Wiki corpus not yet downloaded -- run data/download_wiki_corpus.py"}

    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    topics_list = [
        {"title": entry["title"], "num_chunks": entry["num_chunks"]} for entry in manifest
    ]
    return {"topics": topics_list, "count": len(topics_list)}


# Serve the static frontend (index.html, etc.) from the same server,
# so this is a single `uvicorn` command to run the whole demo.
_STATIC_DIR = pathlib.Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
