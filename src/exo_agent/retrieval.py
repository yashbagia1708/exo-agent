"""
Hybrid retrieval over the wiki_chunks table: combines lexical (Postgres
full-text search) and semantic (pgvector cosine similarity) results via
Reciprocal Rank Fusion (RRF) -- the standard technique used by
production hybrid-search systems (Elasticsearch, Azure AI Search,
Weaviate all implement some form of this).

Why hybrid, not just one or the other:
  - Lexical search is precise for exact terminology a question might use
    verbatim ("transit method", "TRAPPIST-1") but misses paraphrases.
  - Semantic search finds conceptually related passages even when the
    wording differs entirely, but can occasionally surface a tangentially
    related passage over an exact keyword match.
  - RRF takes each result's RANK (not raw score) in each list and combines
    them, which sidesteps the problem of lexical and semantic scores being
    on completely different, incomparable scales.

RRF formula: for each document d, score(d) = sum over each ranked list L
of 1 / (k + rank_L(d)), where k is a small constant (60 is the commonly
cited default) that controls how much weight low ranks still get.
"""

import json
import pathlib
import urllib.error
import urllib.request
from dataclasses import dataclass

import psycopg2
import psycopg2.extras

from exo_agent import config

EMBED_MODEL = "nomic-embed-text"
RRF_K = 60  # standard default constant for RRF


@dataclass
class RetrievedChunk:
    title: str
    source_url: str
    content: str
    rrf_score: float
    lexical_rank: int | None = None
    semantic_rank: int | None = None


def _embed_query(text: str) -> list[float]:
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
            f"Could not reach Ollama's embedding endpoint. Is Ollama running, "
            f"and have you pulled '{EMBED_MODEL}'? Original error: {e}"
        )


def _lexical_search(cur, query: str, limit: int) -> list[tuple]:
    """Returns [(id, title, source_url, content), ...] ranked by full-text relevance."""
    cur.execute(
        """
        SELECT id, title, source_url, content
        FROM wiki_chunks
        WHERE search_vec @@ to_tsquery('english', %s)
        ORDER BY ts_rank(search_vec, to_tsquery('english', %s)) DESC
        LIMIT %s
        """,
        (_to_tsquery_safe(query), _to_tsquery_safe(query), limit),
    )
    return cur.fetchall()


def _to_tsquery_safe(query: str) -> str:
    """
    Converts a free-text question into a basic OR-joined tsquery so it
    doesn't blow up on punctuation or stopword-only input. This is a
    simple approach -- plainto_tsquery would be more robust, but using
    to_tsquery with explicit OR joins keeps individual term matches
    visible for debugging the trace output.
    """
    words = [w.strip() for w in query.split() if w.strip().isalnum()]
    if not words:
        return "exoplanet"  # safe fallback so the query never errors on punctuation-only input
    return " | ".join(words)


def _semantic_search(cur, query_embedding: list[float], limit: int) -> list[tuple]:
    """Returns [(id, title, source_url, content), ...] ranked by cosine similarity."""
    cur.execute(
        """
        SELECT id, title, source_url, content
        FROM wiki_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_embedding, limit),
    )
    return cur.fetchall()


def hybrid_search(question: str, top_k: int = 5, candidate_pool: int = 15) -> list[RetrievedChunk]:
    """
    Run both lexical and semantic search, merge via RRF, return the
    top_k fused results.

    candidate_pool controls how many results each individual search
    contributes before fusion -- a larger pool gives RRF more to work
    with, at the cost of a bit more compute.
    """
    conn = psycopg2.connect(config.DB_DSN)
    try:
        cur = conn.cursor()

        lexical_rows = _lexical_search(cur, question, candidate_pool)
        query_embedding = _embed_query(question)
        semantic_rows = _semantic_search(cur, query_embedding, candidate_pool)
    finally:
        conn.close()

    # Build rank maps: chunk_id -> rank (1-indexed, best = rank 1)
    lexical_ranks = {row[0]: i + 1 for i, row in enumerate(lexical_rows)}
    semantic_ranks = {row[0]: i + 1 for i, row in enumerate(semantic_rows)}

    # Keep metadata from whichever list we saw the chunk in first
    chunk_metadata = {}
    for row in lexical_rows + semantic_rows:
        chunk_id, title, source_url, content = row
        if chunk_id not in chunk_metadata:
            chunk_metadata[chunk_id] = (title, source_url, content)

    all_ids = set(lexical_ranks) | set(semantic_ranks)

    scored = []
    for chunk_id in all_ids:
        rrf_score = 0.0
        if chunk_id in lexical_ranks:
            rrf_score += 1.0 / (RRF_K + lexical_ranks[chunk_id])
        if chunk_id in semantic_ranks:
            rrf_score += 1.0 / (RRF_K + semantic_ranks[chunk_id])

        title, source_url, content = chunk_metadata[chunk_id]
        scored.append(
            RetrievedChunk(
                title=title,
                source_url=source_url,
                content=content,
                rrf_score=rrf_score,
                lexical_rank=lexical_ranks.get(chunk_id),
                semantic_rank=semantic_ranks.get(chunk_id),
            )
        )

    scored.sort(key=lambda c: c.rrf_score, reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "How does the transit method work?"
    print(f"Question: {question}\n")

    results = hybrid_search(question, top_k=5)
    for i, chunk in enumerate(results):
        print(f"#{i + 1} [{chunk.title}] rrf={chunk.rrf_score:.4f} "
              f"(lexical_rank={chunk.lexical_rank}, semantic_rank={chunk.semantic_rank})")
        print(f"   {chunk.content[:160]}...")
        print()
