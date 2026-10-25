# exo-agent

A natural-language-to-insight agent over real NASA exoplanet data. Ask a
question in plain English; it decomposes compound questions, routes each
part to a live PostgreSQL database and/or a hybrid Wikipedia search,
verifies named entities against real data instead of guessing, decides
whether a chart would help, and answers in plain English — grounded
strictly in what the data and retrieved sources actually show.

Built to learn agentic AI patterns properly: multi-step decomposition,
tool routing, RAG/hybrid retrieval, tool-call repair loops, structured
outputs, and an evaluation-first mindset, rather than a single
prompt-and-pray chatbot.

```
"List the planets orbiting TRAPPIST-1, and what is the Kepler space telescope?"
  -> splits into 2 independent sub-questions
  -> sub-question 1: verifies "TRAPPIST-1" against the real database,
     queries it, returns all 7 known planets
  -> sub-question 2: hybrid search finds the real Wikipedia article,
     no SQL involved
  -> "The planets orbiting TRAPPIST-1 are b, c, d, e, f, g, and h.
      Separately, the Kepler space telescope was a NASA observatory
      that operated from 2009 to 2018..." (with a cited source link)
```

## Why this exists

I work as a Data Analyst, and I kept finding myself reaching for AI and
automation beyond what the role strictly asked for — automating manual
ETL steps, building an internal NL-to-SQL assistant, that kind of thing.
This project is the next step: building a genuinely agentic system from
scratch, with the discipline (evaluation, repair loops, grounding) that
separates "a chatbot that demos well" from a system you can actually
trust.

It's deliberately built on a topic I'm personally curious about — NASA's
real exoplanet catalogue — rather than another retail/Titanic dataset.

## Architecture

The agent decomposes the input into 1–N independent sub-questions, runs
each through its own routing/tool/answer pipeline, then synthesizes one
final answer (or passes a single sub-answer straight through, with zero
extra LLM cost, when there's nothing to combine):

```
question
   │
   v
planner ──► decomposes into 1-N sub-questions, each independently routed
   │         to SQL, retrieval, or both
   │
   ├── sub-question 1 ──► [if needs_sql]  entity_resolver → sql_writer →
   │                        executor → chart_decider → [chart_generator]
   │                      [if needs_retrieval]  retriever (hybrid search)
   │                                                       │
   │                                                       v
   │                                              insight_writer
   │                                          (grounded sub-answer,
   │                                           sources, caveat)
   │
   ├── sub-question 2 ──► (same per-sub-question pipeline)
   │
   └── sub-question N ──► ...
                              │
                              v
                       synthesizer ──► sub-answers stitched together verbatim
                       (skipped — passthrough — if there was only
                        1 sub-question, the common case)
```

Implemented as a real [LangGraph](https://github.com/langchain-ai/langgraph)
`StateGraph` (`src/exo_agent/graph.py`) — an outer graph runs the
planner/synthesizer, and an inner compiled sub-graph runs once per
sub-question. Chart generation and tool routing are genuine conditional
edges, not if-statements hidden inside a node. Sub-questions run
sequentially (not LangGraph's parallel `Send` fan-out) — a deliberate
choice since most real input is 1 sub-question anyway, sequential output
stays readable in trace order, and a single local Ollama instance can't
meaningfully parallelize multiple concurrent calls regardless.

### The two knowledge sources

- **Structured data** — real SQL against a live PostgreSQL database of
  6,298 confirmed exoplanets. Used for counts, comparisons, trends,
  named-entity lookups.
- **Unstructured knowledge** — hybrid search (lexical full-text +
  semantic vector search, fused via Reciprocal Rank Fusion) over 18
  curated Wikipedia articles on detection methods and astronomy concepts.
  Used for "what is X" / "how does X work" questions with no column in
  the database.

The planner decides per-sub-question which source(s) a question actually
needs — this is the agent's real routing/tool-selection behavior, not a
fixed pipeline that always runs both.

## What makes this more than a wrapper around an LLM

- **Grounded, not memorized.** The agent never states a number or fact
  from memory — every number traces back to a real SQL row, every
  concept traces back to a cited, retrieved Wikipedia passage.
- **Verifies named entities instead of guessing them.** When a question
  names a specific planet/star, `entity_resolver.py` looks it up against
  the real database using Postgres's `pg_trgm` trigram similarity —
  deterministic string matching, not an LLM guess at catalogue naming
  conventions (e.g. correctly resolving "Proxima Centauri b" to the real
  stored value "Proxima Cen b"). See FINDINGS.md for the bug this fixed.
- **Two repair loops.** A lightweight structural validator catches
  malformed SQL (truncated CTEs, unbalanced parens) before it ever
  reaches Postgres; the executor catches *semantic* errors (wrong types,
  bad joins) using Postgres's own real error message, and retries with
  that error fed back to the model.
- **Decomposes compound questions.** A message joining two unrelated
  asks ("list X, and also tell me Y") is split into independent
  sub-questions before any SQL is written — fixing a real bug where a
  fused query silently answered neither part correctly.
- **Provider-agnostic by design.** Every LLM call goes through one
  `instructor`-wrapped client (`src/exo_agent/providers/factory.py`).
  Swapping between local Ollama models and Claude/OpenAI is a one-line
  config change (`PROVIDER=ollama` vs `PROVIDER=anthropic`), with zero
  changes to any node's logic.
- **Evaluated, not vibe-checked.** `eval/run_eval.py` runs each
  hand-written test question 5 times and reports a *pass rate*, not a
  single pass/fail — because LLM behavior on identical input isn't
  deterministic, and a single clean test run can be misleading (see
  FINDINGS.md for a real, concrete example of exactly this).
- **A live, inspectable demo.** The frontend (`frontend/`) shows a
  glanceable routing summary (which sub-question went to the database vs.
  Wikipedia, with the real SQL/sources) and a full step-by-step trace —
  built from the agent's actual execution state, not a separate,
  prettified fiction of what happened.

## results

Run against `qwen2.5-coder:14b` (local, free, via Ollama):

| Tier | What it tests | Pass rate |
|---|---|---|
| 1 — basic lookups | Simple counts, filters, ORDER BY | 22/25 (88%) |
| 2 — joins & aggregation | Multi-table joins, GROUP BY | 17/20 (85%) |
| 3 — time trends | Chart-worthy trend questions | 10/10 (100%) |
| 5 — limits & refusal | Should the agent decline/caveat? | 10/15 (67%) |

The tier 5 number is the most important one, and it's the one I'd flag
first in an interview rather than hide: **the question "which planet has
confirmed life on it" was only correctly declined in 1 of 5 runs.**
Questions with a real proxy column to fall back on (e.g. "surface
temperature" → equilibrium temperature) are handled fairly reliably (4/5).
Questions with *no* answerable basis at all in the dataset are handled
much less reliably. That's a genuine, current limitation of this system,
not a solved problem — seeing this clearly is the actual point of having
an eval harness instead of just trying a few demo questions and feeling
good about it.

**A second, more serious limitation, found after adding multi-question
decomposition:** when a compound question splits into 3+ sub-questions
and one of them asks for something genuinely outside the schema (e.g.
"which planet has the highest radial velocity" — there's no measured
velocity column, only a detection-method label), the final synthesis
step has, on different runs, both fabricated a vague "as of my training
data" disclaimer *and*, after a prompt fix for that, flatly contradicted
a correct, populated sub-answer (claiming "no planets were found" when
the actual SQL had just returned 200 real rows). Two prompt-level fixes
in a row did not reliably close this gap — full detail, including the
reasoning for not yet attempting a third prompt patch, is in
[`FINDINGS.md`](./FINDINGS.md)'s most recent entries. This is currently
an open, unresolved finding, not something fixed and verified — I'd
rather say that plainly than imply a fix landed when the evidence so far
says it only partially did.

Full breakdown, including *why* particular failures happened and what
was tried, is in [`FINDINGS.md`](./FINDINGS.md) — a dated log covering
distinct failure categories found during development: instruction
overreach, structural SQL truncation, semantic type errors, silent
NULL-as-extremum bugs, unreliable refusal behavior, entity-name guessing
vs. deterministic verification, a physical-reasoning direction error, and
the still-open synthesizer grounding gap described above.

## Model comparison

Also compared `llama3.1:8b` (general-purpose) against
`qwen2.5-coder:14b` (code-specialized) on identical questions.
`qwen2.5-coder:14b` converged correctly on every case tested, including
one where `llama3.1:8b` oscillated between two wrong SQL fixes without
ever finding the right one. Full comparison in `FINDINGS.md` — this is
why `qwen2.5-coder:14b` is the default model, backed by measured evidence
rather than a guess.

The provider/model is swappable at zero code cost:

```bash
PROVIDER=ollama MODEL=qwen2.5-coder:14b python3 -m exo_agent.graph "..."
PROVIDER=anthropic MODEL=claude-sonnet-4-6 python3 -m exo_agent.graph "..."
```

## The data

[NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/),
PSCompPars table — 6,298 confirmed exoplanets across 4,716 unique host
stars, downloaded directly via NASA's public TAP API
(`data/download_data.py`) and split into a two-table relational schema
(`data/load_to_postgres.py`).

Real astronomy data comes with real messiness: stellar parameters
sometimes disagree slightly across a star's planet rows (different
literature references), and several columns have genuine missing
measurements rather than zeros. Both are documented honestly in
[`data/data_quality_notes.md`](./data/data_quality_notes.md), generated
automatically by the loader, not hand-written after the fact.

## Project structure

```
exo-agent/
├── README.md, FINDINGS.md          ← start here: overview + the honest bug log
│
├── src/exo_agent/                  ← the actual agent package
│   ├── graph.py                     LangGraph wiring: decomposer → per-sub-question
│   │                                pipeline → synthesizer (see Architecture above)
│   ├── config.py                    provider/model selection, DB connection
│   ├── schemas.py                   every Pydantic structured-output model
│   ├── schema_context.py            real DB schema description fed to the LLM,
│   │                                including every worked-example fix from FINDINGS.md
│   ├── sql_validator.py             structural SQL validation, pre-execution
│   ├── retrieval.py                 hybrid search: lexical + semantic, RRF-fused
│   ├── providers/
│   │   └── factory.py               provider-agnostic client (Ollama/Claude/OpenAI)
│   └── nodes/                       one file per pipeline stage
│       ├── planner.py                decomposes input, routes each sub-question
│       ├── entity_resolver.py        verifies named planets/stars via pg_trgm
│       │                             similarity (not LLM guessing)
│       ├── sql_writer.py             NL → validated SQL
│       ├── executor.py               runs SQL, repairs on real Postgres errors
│       ├── chart_decider.py          decides chart type from actual result shape
│       ├── chart_generator.py        matplotlib rendering, no LLM call
│       └── insight_writer.py         grounded answer + caveat + cited sources
│
├── data/                            ← everything that builds the two knowledge sources
│   ├── download_data.py              pulls real planet data from NASA's public API
│   ├── load_to_postgres.py           builds the host_stars + planets schema
│   ├── data_quality_notes.md         auto-generated data quality findings
│   ├── download_wiki_corpus.py       pulls ~18 curated Wikipedia articles
│   ├── load_wiki_to_postgres.py      embeds + loads them into wiki_chunks (pgvector)
│   ├── RAG_SETUP.md                  step-by-step setup for the retrieval layer
│   └── README.md                     step-by-step setup for the structured-data layer
│
├── eval/                            ← the evaluation harness
│   ├── questions.jsonl               hand-written test questions across 5 difficulty tiers
│   └── run_eval.py                   runs each question 5x, reports PASS RATES
│
├── frontend/                        ← the live demo UI
│   ├── server.py                     FastAPI backend — calls graph.py directly,
│   │                                 no separate/simplified agent logic
│   ├── static/index.html             single-page UI: routing summary, full trace,
│   │                                 sources, charts, topic browser
│   └── README.md                     how to run the demo
│
├── tests/                           ← deterministic unit tests, run by CI
├── docker-compose.yml, docker/      ← local Postgres with pgvector + pg_trgm
├── Makefile                         ← setup / load / serve / test shortcuts
│
├── launch.py                        ← one-command launcher (python3 launch.py)
├── compare_models.py                side-by-side local-model comparison tool
├── test_provider.py                  quick smoke test for the provider layer
└── PROJECT_CONTEXT.md               compact project-state summary for resuming
                                      work in a fresh chat session
```

**Reading order, if you're new to this repo:** `README.md` (this file) →
the architecture diagram above → `FINDINGS.md` for the real bugs and
fixes → `src/exo_agent/graph.py` for the actual wiring → any individual
`nodes/*.py` file for a specific stage's logic.

## Running it

Needs Python 3.10+, Docker, and [Ollama](https://ollama.com). Everything
runs locally and free. One command starts it all:

```bash
python3 launch.py     # or: make demo
```

The launcher creates `.venv`, starts Postgres in Docker and the Ollama
server, pulls the models, loads the data on first run, then opens the UI
at http://localhost:8420. Anything already running is reused, and Ctrl+C
stops what it started. The first run downloads about 9 GB of models;
after that it's up in seconds.

To run the steps yourself instead:

```bash
make setup        # .venv + dependencies, and .env copied from .env.example
make db-up        # Postgres 16 + pgvector + pg_trgm in Docker, on localhost:5433
ollama serve      # in its own terminal
make models       # pulls qwen2.5-coder:14b and nomic-embed-text
make load-data    # downloads NASA's planet table, loads it, embeds the wiki corpus
```

Then:

```bash
.venv/bin/python -m exo_agent.graph "How many confirmed exoplanets are there?"
make serve        # live demo UI at http://localhost:8420
make test         # deterministic unit tests (no LLM or database), same as CI
make eval         # LLM eval harness: each question 5x, reports pass rates (slow)
```

`make help` lists every target. To use Claude or OpenAI instead of
Ollama, set `PROVIDER` and the matching API key in `.env`.

If you'd rather use an existing Postgres than Docker, point `DB_DSN` in
`.env` at it; `data/README.md` and `data/RAG_SETUP.md` cover installing
the extensions by hand.

## What I'd build next

- **Top priority:** resolve the open synthesizer grounding gap above —
  either confirm/rule out a model-capability ceiling by testing the exact
  repro on Claude, or redesign the synthesizer to be more mechanical
  (template directly from sub-answer data) rather than freely generated,
  trading some fluency for a much stronger grounding guarantee
- Add cross-sub-question dependency handling — a sub-question like "how
  does *this* planet compare to Earth" currently can't reference an
  earlier sub-question's resolved entity/result, since each sub-question
  runs independently
- Investigate the tier-5 refusal reliability gap (1/5 on the life-detection
  question) properly, rather than just measuring it
- Add observability (Langfuse) for real token-cost and latency tracking
  across runs, not just wall-clock timing
- Extend the agent's tool set beyond SQL/retrieval — e.g. a real
  web-search tool for questions neither the database nor the Wikipedia
  corpus can answer, with the agent deciding when to reach for it
