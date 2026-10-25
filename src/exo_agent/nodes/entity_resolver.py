"""
Entity resolver: find the stored name for a planet/star the question mentions.

    "Proxima Centauri b" -> "Proxima Cen b"  (planets, similarity 0.6)
    "Zorblax Prime"      -> no match          (probably not in the data)

The model only pulls out the name as written; the match itself is a
pg_trgm similarity query, so the SQL writer gets a real value instead of
guessing catalogue abbreviations with ILIKE.
"""

from dataclasses import dataclass

import psycopg2

from exo_agent import config
from exo_agent.providers.factory import get_client
from exo_agent.schemas import EntityExtraction

SIMILARITY_THRESHOLD = 0.35  # pg_trgm default is 0.3; a bit stricter to skip weak matches

EXTRACTION_SYSTEM_PROMPT = """Extract any specific named planet or star \
mentioned in this question, if any. Return the name as it appears in the \
question -- do not normalize, abbreviate, or correct it yourself. If no \
specific named planet or star is mentioned (the question is about counts, \
categories, or trends with no specific name), return an empty string.
"""


@dataclass
class ResolvedEntity:
    mentioned_as: str
    resolved_name: str | None  # stored value, or None if nothing close enough
    table: str | None  # 'planets' or 'host_stars'
    similarity_score: float | None


def _extract_entity_mention(question: str) -> str:
    client = get_client()
    result = client.chat.completions.create(
        model=config.MODEL,
        response_model=EntityExtraction,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    return result.entity_name.strip()


def _trigram_lookup(mentioned_name: str) -> ResolvedEntity:
    """Best pg_trgm match across pl_name and hostname. No LLM here."""
    conn = psycopg2.connect(config.DB_DSN)
    try:
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        conn.commit()

        cur.execute(
            """
            SELECT pl_name AS name, 'planets' AS source_table,
                   similarity(pl_name, %(q)s) AS score
            FROM planets
            UNION ALL
            SELECT hostname AS name, 'host_stars' AS source_table,
                   similarity(hostname, %(q)s) AS score
            FROM host_stars
            ORDER BY score DESC
            LIMIT 1
            """,
            {"q": mentioned_name},
        )
        row = cur.fetchone()
    finally:
        conn.close()

    if row is None or row[2] < SIMILARITY_THRESHOLD:
        return ResolvedEntity(
            mentioned_as=mentioned_name,
            resolved_name=None,
            table=None,
            similarity_score=row[2] if row else None,
        )

    name, table, score = row
    return ResolvedEntity(
        mentioned_as=mentioned_name, resolved_name=name, table=table, similarity_score=score
    )


def resolve_entity(question: str) -> ResolvedEntity | None:
    """None if no planet/star is named (most questions), else the best match."""
    mentioned = _extract_entity_mention(question)
    if not mentioned:
        return None
    return _trigram_lookup(mentioned)


if __name__ == "__main__":
    import sys

    test_questions = [
        "Tell me the equilibrium temperature of Proxima Centauri b",
        "Which planets orbit TRAPPIST-1?",
        "How many confirmed exoplanets are there?",
        "What's the mass of Kepler-22b?",
        "Tell me about a planet called Zorblax Prime",  # made up -> no match
    ]

    questions = sys.argv[1:] if len(sys.argv) > 1 else test_questions

    for q in questions:
        print(f"\nQuestion: {q}")
        result = resolve_entity(q)
        if result is None:
            print("  No named entity mentioned -- nothing to resolve.")
        elif result.resolved_name is None:
            print(f"  Mentioned '{result.mentioned_as}', but no confident match found "
                  f"(best similarity: {result.similarity_score:.3f} < {SIMILARITY_THRESHOLD}).")
        else:
            print(f"  Mentioned '{result.mentioned_as}' -> resolved to real "
                  f"'{result.resolved_name}' in {result.table} (similarity: {result.similarity_score:.3f})")
