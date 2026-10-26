# exo-agent — Project Context (compact summary for new chat)

## What this is
A portfolio project for a PwC "AI Engineer / Data Scientist (Agentic AI)"
job application (Associate/Senior Associate level, Sydney). Built to
demonstrate agentic AI engineering skills the job listing names directly:
multi-step agents, tool routing, RAG/hybrid retrieval, structured outputs,
tool-call repair loops, and an evaluation-first mindset. Already got an
interview invite based on resume + portfolio; now polishing the actual
project before/around the interview.

## Architecture
LangGraph agent, two-stage:
```
question → planner (decomposes into N sub-questions, routes each to
           SQL and/or retrieval) → for each sub-question, run inner
           pipeline → synthesizer (if >1 sub-question: stitches the
           sub-answers verbatim, NO LLM call) → final answer
```
Inner per-sub-question pipeline:
```
[sql_writer → executor → chart_decider → chart_generator] and/or
[retriever (hybrid search)]
  → insight_writer (grounded answer + caveat + sources)
```

## Stack
- **Data:** NASA Exoplanet Archive (PSCompPars), 6,451 planets / 4,847
  host stars (Oct 2026 download; the June eval runs used 6,298 / 4,716),
  real download via TAP API → Postgres (`planets`, `host_stars` tables)
- **RAG corpus:** 18 Wikipedia articles (detection methods, astronomy
  concepts) → `wiki_chunks` table with pgvector embeddings + tsvector
  full-text, hybrid search via Reciprocal Rank Fusion (RRF_K=60)
- **LLM:** Ollama local, default model `qwen2.5-coder:14b` (chosen via
  real comparison vs llama3.1:8b — see FINDINGS.md). Embeddings via
  `nomic-embed-text` (768-dim). Provider-agnostic via `instructor` +
  factory pattern — Claude/OpenAI swap-in via `PROVIDER` env var, but
  NOT YET TESTED on Claude (cost decision, $5 min top-up not spent yet)
- **Structured outputs:** Pydantic schemas throughout (SQLQuery,
  ExecutionResult, ChartSpec, Insight, RoutingPlan/DecompositionPlan,
  SubQuestion, ResolvedEntity)
- **Entity resolution:** pg_trgm trigram similarity — verifies named
  planets/stars against real DB values BEFORE SQL writing (not LLM
  guessing) — fixes "Proxima Centauri" vs real "Proxima Cen" mismatches
- **Frontend:** FastAPI backend (`frontend/server.py`) + single-page HTML
  (`frontend/static/index.html`), no build step. `/api/ask/stream` sends
  server-sent events so the trace fills in step by step as each graph
  node finishes; `/api/ask` returns one response and is the fallback.
  Shows the plan, live steps, answer, caveat, sources, chart and SQL
- **Infra:** Postgres 16 + pgvector + pg_trgm in Docker
  (`docker-compose.yml`, host port 5433). `python3 launch.py` (or
  `make demo`) starts everything: .venv, Docker DB, Ollama, models, data
  load on first run, web UI. GitHub Actions CI runs ruff + pytest
- **Eval:** `eval/run_eval.py` — runs each question 5x, reports PASS
  RATES not pass/fail (non-determinism matters — proven empirically).
  Deterministic unit tests (no LLM/DB) live in `tests/` and run in CI

## Project structure
```
exo-agent/
├── README.md, FINDINGS.md          (main docs — FINDINGS.md has 22 dated entries)
├── launch.py, Makefile             (one-command launcher, task shortcuts)
├── docker-compose.yml, docker/     (Postgres + extensions)
├── .github/workflows/ci.yml        (ruff + pytest on every push)
├── src/exo_agent/
│   ├── graph.py                     (LangGraph wiring — planner/decompose/synthesize)
│   ├── config.py, schemas.py, schema_context.py, sql_validator.py, retrieval.py
│   └── nodes/ (planner, sql_writer, executor, chart_decider, chart_generator,
│               entity_resolver, insight_writer)
├── data/ (download_data.py, load_to_postgres.py, download_wiki_corpus.py,
│          load_wiki_to_postgres.py, data_quality_notes.md, RAG_SETUP.md)
├── eval/ (questions.jsonl — 20 questions across 5 tiers, run_eval.py)
├── tests/ (test_sql_validator.py, test_synthesizer.py)
└── frontend/ (server.py, static/index.html)
```

## Current known state / open issues (see FINDINGS.md for full detail)
1. **FIXED (2026-10-26):** Synthesizer fabrication ("knowledge cutoff",
   then contradicting a 200-row result). Rebuilt as a verbatim stitch with
   no LLM call; 4 regression tests in `tests/test_synthesizer.py`. Trade-off:
   compound answers read as separate paragraphs, and near-duplicate
   sub-answers repeat.
2. Tier 5 eval (refusal on unanswerable questions like "which planet has
   life") only 1/5 pass rate — known, documented, honest limitation. Now
   the top open priority.
3. **Much improved, not solved:** radial velocity. No measured RV column
   exists (only discoverymethod='Radial Velocity'). The real culprit was
   insight_writer, not the SQL prompt; a worked example there took false
   "highest RV" answers from 5/5 to 0/5 (caveat 5/5), but 1/5 runs still
   claimed a planet "has a radial velocity measurement" while its own
   caveat said otherwise. Not yet in eval/questions.jsonl.
4. Cross-sub-question dependencies (e.g. "how does THIS compare to
   Earth" referring to a previous sub-question's answer) not supported —
   each sub-question runs independently. Known architectural gap.
5. insight_writer only sees the first 30 rows, so a long list (e.g. 200
   planets from 2025) shows 30 names without saying it's truncated.
6. Fixed in June: data was split across TWO Postgres databases (`exoagent`
   vs `postgres`) due to mid-debugging `DB_DSN` export that didn't
   persist across terminal sessions — caused retrieval to silently find
   nothing. Fixed by setting DB_DSN explicitly in `.env`. Also made
   retriever_node error logging louder (logs exception type, not just
   message) and surfaced retrieval errors in the frontend trace.

## Environment notes (yours specifically)
- Mac, Apple Silicon (arm64) — had recurring Rosetta (x86_64 emulation)
  issues silently breaking compilers/tools; fixed by checking terminal
  app's "Open using Rosetta" setting
- Conda env at `/Users/yashbagia/Public/anaconda3` — `python3` alone
  often resolves to wrong interpreter; use full path or `pip install -e .`
- DB connection: `.env` now points at the Docker database,
  `DB_DSN=dbname=exoagent user=exoagent password=exoagent host=localhost
  port=5433`, which has all 3 tables (planets, host_stars, wiki_chunks).
  The old Homebrew Postgres (`dbname=postgres` on port 5432, with the
  June data) still exists; its line is commented out in `.env`
- History: pgvector once had to be built from source against Homebrew
  Postgres 16; the Docker image (`pgvector/pgvector:pg16`) avoids that
- Ollama models pulled: `qwen2.5-coder:14b` (primary), `llama3.1:8b`
  (comparison baseline), `nomic-embed-text` (embeddings)

## How to resume
Run `python3 launch.py` (or `make demo`) from the project root: it starts
the Docker database and Ollama if needed and opens the UI at
http://localhost:8420. For the CLI, `.venv/bin/python -m exo_agent.graph
"<question>"` (Postgres and Ollama must be running). `make test` runs the
unit tests; `make eval` runs the 5x LLM eval.

## Working style established in this project
- Every fix gets logged in FINDINGS.md with: what happened, root cause,
  fix applied, and HONEST status (don't claim "fixed" until live-verified)
- Prefer the smallest defensible fix; escalate to bigger architectural
  changes only when prompt-only fixes demonstrably plateau
- When something doesn't work, diagnose with direct/isolated tests
  (test the specific function, not just the full pipeline) before
  guessing at fixes
- Always syntax-check files before sending; catch own mistakes (this has
  happened a few times — duplicate code from bad edits, variable
  shadowing in JS — always re-verify after editing)
