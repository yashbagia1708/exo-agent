"""
Loads the downloaded Wikipedia corpus (data/wiki_corpus/*.json) into
Postgres with BOTH a vector embedding column (pgvector, for semantic
search) and a full-text search column (tsvector, for lexical search) --
the two halves of hybrid retrieval.

Embeddings are generated via Ollama's nomic-embed-text model -- free,
local, no API key needed, the same pattern as the chat models already
used elsewhere in this project.

Requires the pgvector extension. On most systems:
    psql exoagent -c "CREATE EXTENSION IF NOT EXISTS vector;"
This script attempts that itself, but you may need superuser privileges
locally if it's not already installed -- see the error message if it
fails.
"""

import json
import pathlib
import sys
import urllib.error
import urllib.request

import psycopg2
import psycopg2.extras

from exo_agent import config

WIKI_CORPUS_DIR = pathlib.Path(__file__).parent / "wiki_corpus"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768  # nomic-embed-text's native output dimension

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS wiki_chunks;

CREATE TABLE wiki_chunks (
    id          SERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    source_url  TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    embedding   VECTOR(%(dim)s),
    search_vec  TSVECTOR
);

CREATE INDEX idx_wiki_chunks_search ON wiki_chunks USING GIN(search_vec);
"""

# A separate statement so we can format the dimension safely without
# psycopg2 trying to parameterize inside a DDL VECTOR(...) type spec.
SCHEMA_SQL_FORMATTED = SCHEMA_SQL % {"dim": EMBED_DIM}

UPDATE_SEARCH_VEC_SQL = """
UPDATE wiki_chunks SET search_vec = to_tsvector('english', content);
"""

# pgvector's ANN index (HNSW) is built AFTER data is loaded, since
# building it on an empty table is pointless and building it incrementally
# during inserts is slower than a single bulk build at the end.
CREATE_VECTOR_INDEX_SQL = """
CREATE INDEX idx_wiki_chunks_embedding
    ON wiki_chunks USING hnsw (embedding vector_cosine_ops);
"""


def get_embedding(text: str) -> list[float]:
    """
    Call Ollama's embedding endpoint directly via urllib -- deliberately
    NOT going through the instructor/chat-completion client used
    elsewhere, since embeddings are a different API shape (no structured
    output, no chat messages, just text in -> vector out).
    """
    payload = json.dumps({"model": EMBED_MODEL, "input": text}).encode("utf-8")
    req = urllib.request.Request(
        f"{config.OLLAMA_BASE_URL.replace('/v1', '')}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["embeddings"][0]
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama's embedding endpoint at {req.full_url}. "
            f"Is Ollama running, and have you pulled '{EMBED_MODEL}'? "
            f"(ollama pull {EMBED_MODEL})  Original error: {e}"
        )


def load_corpus_files() -> list[dict]:
    if not WIKI_CORPUS_DIR.exists():
        print(
            f"ERROR: {WIKI_CORPUS_DIR} not found. Run download_wiki_corpus.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    records = []
    for path in sorted(WIKI_CORPUS_DIR.glob("*.json")):
        if path.name == "_manifest.json":
            continue
        records.append(json.loads(path.read_text(encoding="utf-8")))

    if not records:
        print(f"ERROR: no corpus files found in {WIKI_CORPUS_DIR}.", file=sys.stderr)
        sys.exit(1)

    return records


def main():
    records = load_corpus_files()
    total_chunks = sum(len(r["chunks"]) for r in records)
    print(f"Loaded {len(records)} article(s), {total_chunks} chunk(s) total.")

    try:
        conn = psycopg2.connect(config.DB_DSN)
    except psycopg2.OperationalError as e:
        print(f"ERROR: could not connect to Postgres: {e}", file=sys.stderr)
        sys.exit(1)

    cur = conn.cursor()

    print("\nCreating schema (requires pgvector extension)...")
    try:
        cur.execute(SCHEMA_SQL_FORMATTED)
        conn.commit()
    except psycopg2.Error as e:
        conn.rollback()
        print(
            f"ERROR creating schema: {e}\n"
            "If this is about the 'vector' extension, you may need to install "
            "pgvector first (e.g. `brew install pgvector` on macOS, then restart "
            "Postgres), then retry.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Embedding and inserting chunks (this calls Ollama once per chunk -- may take a bit)...")
    inserted = 0
    for record in records:
        for i, chunk in enumerate(record["chunks"]):
            try:
                embedding = get_embedding(chunk)
            except RuntimeError as e:
                print(f"  ERROR embedding chunk {i} of '{record['title']}': {e}", file=sys.stderr)
                conn.rollback()
                sys.exit(1)

            cur.execute(
                """
                INSERT INTO wiki_chunks (title, source_url, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s, %s::vector)
                """,
                (record["title"], record["source_url"], i, chunk, embedding),
            )
            inserted += 1

        print(f"  {record['title']}: {len(record['chunks'])} chunk(s) inserted")

    conn.commit()
    print(f"\nInserted {inserted} chunk(s) total.")

    print("Building full-text search vectors...")
    cur.execute(UPDATE_SEARCH_VEC_SQL)
    conn.commit()

    print("Building vector index (HNSW, this may take a moment)...")
    try:
        cur.execute(CREATE_VECTOR_INDEX_SQL)
        conn.commit()
    except psycopg2.Error as e:
        conn.rollback()
        print(
            f"  WARNING: could not build HNSW index ({e}). "
            "Vector search will still work, just without the index speedup -- "
            "fine at this corpus size (a few hundred chunks).",
            file=sys.stderr,
        )

    cur.execute("SELECT count(*) FROM wiki_chunks;")
    final_count = cur.fetchone()[0]
    print(f"\nDone. wiki_chunks table has {final_count} row(s).")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
