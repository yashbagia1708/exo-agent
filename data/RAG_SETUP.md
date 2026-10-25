# RAG / Hybrid Search — setup

Adds a second knowledge source on top of the structured planets database:
hybrid (lexical + semantic) search over a small corpus of Wikipedia
articles covering exoplanet detection methods and astronomy concepts.

## Why this exists

The structured database answers "how many" / "which" / "compare"
questions well, but has no way to answer "how does the transit method
work" or "what is a habitable zone" -- there's no column for that. This
layer covers exactly that gap, and only that gap (it's not a general
Wikipedia search engine; the corpus is a curated ~20 pages chosen to
complement the structured data, not duplicate it).

## 0. Enable pg_trgm (for entity name resolution)

```bash
psql postgres -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"
```

This is a separate extension from pgvector -- `pg_trgm` does deterministic
fuzzy string matching (trigram similarity), used by
`src/exo_agent/nodes/entity_resolver.py` to verify a named planet/star
against the real database BEFORE the SQL writer uses it, instead of
letting the LLM guess at catalogue naming/abbreviation conventions. See
FINDINGS.md for the real bug this fixes (guessing "Proxima Centauri b"
when the real stored name is "Proxima Cen b").

It's created automatically by `entity_resolver.py` on first use too
(`CREATE EXTENSION IF NOT EXISTS` is idempotent), but running it manually
first lets you confirm it's available without errors before relying on it
mid-question.

## 1. Install pgvector (Postgres extension, not a pip package)

**macOS (Homebrew):**
```bash
brew install pgvector
```

If you installed Postgres via Homebrew, this should link automatically.
Restart Postgres after installing:
```bash
brew services restart postgresql@16
```

## 2. Pull the embedding model

```bash
ollama pull nomic-embed-text
```

This is a small (~270MB), free, local embedding model — separate from
your chat model (`qwen2.5-coder:14b`). Both can be pulled and used
side by side; Ollama serves whichever model a request asks for.

## 3. Download the Wikipedia corpus

```bash
python3 data/download_wiki_corpus.py
```

Pulls ~20 curated articles via Wikipedia's public REST API, chunks each
into ~1200-character passages with overlap, and saves them as JSON under
`data/wiki_corpus/`.

## 4. Load into Postgres (embeds + indexes)

```bash
PROVIDER=ollama python3 data/load_wiki_to_postgres.py
```

This creates a `wiki_chunks` table with both a `VECTOR(768)` column
(pgvector, for semantic search) and a `TSVECTOR` column (Postgres
full-text search, for lexical search) — the two halves of hybrid
retrieval live in one table. Each chunk is embedded via Ollama's
`nomic-embed-text` model as it's inserted.

## 5. Test retrieval directly

```bash
PROVIDER=ollama python3 -m exo_agent.retrieval "How does the transit method work?"
```

This runs the real hybrid search (lexical + semantic + RRF fusion) and
prints the top 5 fused results with their individual lexical/semantic
ranks, so you can see exactly how the fusion combined them.

## 6. Test the planner's routing decision

```bash
PROVIDER=ollama python3 -m exo_agent.nodes.planner
```

Runs a few test questions through the planner and shows whether it
correctly routes data questions to SQL, concept questions to retrieval,
or both.

## 7. Run the full agent (now routing-aware)

```bash
PROVIDER=ollama python3 -m exo_agent.graph "How does the transit method work?"
PROVIDER=ollama python3 -m exo_agent.graph "How many planets were found by the transit method?"
PROVIDER=ollama python3 -m exo_agent.graph "How many planets were found by the transit method, and how does that method actually work?"
```

The third question is the interesting one — it should trigger both
branches, and the final answer should clearly distinguish the database
count from the retrieved explanation, with a real Wikipedia source URL
cited.
