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
           pipeline → synthesizer (if >1 sub-question) → final answer
```
Inner per-sub-question pipeline:
```
[sql_writer → executor → chart_decider → chart_generator] and/or
[retriever (hybrid search)]
  → insight_writer (grounded answer + caveat + sources)
```

## Stack
- **Data:** NASA Exoplanet Archive (PSCompPars), 6,298 planets / 4,716
  host stars, real download via TAP API → Postgres (`planets`,
  `host_stars` tables)
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
  SubQuestion, ResolvedEntity, SynthesizedInsight)
- **Entity resolution:** pg_trgm trigram similarity — verifies named
  planets/stars against real DB values BEFORE SQL writing (not LLM
  guessing) — fixes "Proxima Centauri" vs real "Proxima Cen" mismatches
- **Frontend:** FastAPI backend (`frontend/server.py`) + single-page HTML
  (`frontend/static/index.html`) — space-themed UI, shows routing summary
  (which sub-question → SQL/Wikipedia), full step trace, sources, charts
- **Eval:** `eval/run_eval.py` — runs each question 5x, reports PASS
  RATES not pass/fail (non-determinism matters — proven empirically)

## Project structure
```
exo-agent/
├── README.md, FINDINGS.md          (main docs — FINDINGS.md has ~12 dated entries)
├── src/exo_agent/
│   ├── graph.py                     (LangGraph wiring — planner/decompose/synthesize)
│   ├── config.py, schemas.py, schema_context.py, sql_validator.py, retrieval.py
│   └── nodes/ (planner, sql_writer, executor, chart_decider, chart_generator,
│               entity_resolver, insight_writer)
├── data/ (download_data.py, load_to_postgres.py, download_wiki_corpus.py,
│          load_wiki_to_postgres.py, data_quality_notes.md, RAG_SETUP.md)
├── eval/ (questions.jsonl — 14 questions across 5 tiers, run_eval.py)
└── frontend/ (server.py, static/index.html)
```

## Current known state / open issues (see FINDINGS.md for full detail)
1. **OPEN, UNRESOLVED:** Synthesizer (multi-sub-question combiner) has
   fabrication-adjacent bugs — once said "knowledge cutoff 2021" (fixed),
   then on retest CONTRADICTED a correct, populated SQL result (claimed
   "no planets found" when 200 real rows existed). Two prompt-only fixes
   in a row didn't reliably close this. Decision: NOT testing on Claude
   yet (cost), NOT yet rebuilding synthesizer to be mechanical/templated
   instead of LLM-generated. This is the #1 priority next step.
2. Tier 5 eval (refusal on unanswerable questions like "which planet has
   life") only 1/5 pass rate — known, documented, honest limitation.
3. No measured radial velocity column exists in schema (only
   discoverymethod='Radial Velocity' as a label) — fixed via schema_context
   instruction, not yet re-verified live.
4. Cross-sub-question dependencies (e.g. "how does THIS compare to
   Earth" referring to a previous sub-question's answer) not supported —
   each sub-question runs independently. Known architectural gap.
5. Just fixed: data was split across TWO Postgres databases (`exoagent`
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
- pgvector had to be built from source against Postgres 16 specifically
  (Homebrew's bottle was built against 17/18 by default):
  `PG_CONFIG=/opt/homebrew/opt/postgresql@16/bin/pg_config make install`
- DB connection: set in `.env` as `DB_DSN=dbname=postgres user=yashbagia
  host=localhost port=5432` — this is YOUR real working database with
  all 3 tables (planets, host_stars, wiki_chunks)
- Ollama models pulled: `qwen2.5-coder:14b` (primary), `llama3.1:8b`
  (comparison baseline), `nomic-embed-text` (embeddings)

## How to resume
Run `PROVIDER=ollama python3 -m exo_agent.graph "<question>"` from
project root, or start the frontend (`cd frontend && PROVIDER=ollama
uvicorn server:app --reload --port 8420`). Postgres + `ollama serve` must
both be running first (separate terminal for ollama serve).

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
