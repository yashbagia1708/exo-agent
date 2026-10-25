"""
Entity resolver: given a sub-question, extracts any planet/star name it
mentions and looks up real fuzzy matches against the database using
pg_trgm trigram similarity, rather than relying on the LLM to guess.

Catalogue names are often abbreviated or formatted differently than
common usage (e.g. the star commonly called "Proxima Centauri" is
stored as "Proxima Cen" in this dataset). Asking an LLM to use ILIKE
with wildcards on a guessed name still requires it to pick the right
substring, and a full guessed name can fail to match an abbreviated real
value even with wildcards around it.

Instead, this module resolves the name deterministically: a cheap LLM
call extracts the likely name mention, then a real trigram similarity
query against the database returns an actual, measurable score. If a
strong match is found, the SQL writer is handed the confirmed real name
directly and never has to guess. If no good match is found, that's a
reliable signal the named object likely isn't in the dataset.
"""

from dataclasses import dataclass

import psycopg2

from exo_agent import config
from exo_agent.providers.factory import get_client
from exo_agent.schemas import EntityExtraction

SIMILARITY_THRESHOLD = 0.35  # pg_trgm's default is 0.3; slightly stricter to avoid weak matches

EXTRACTION_SYSTEM_PROMPT = """Extract any specific named planet or star \
mentioned in this question, if any. Return the name as it appears in the \
question -- do not normalize, abbreviate, or correct it yourself. If no \
specific named planet or star is mentioned (the question is about counts, \
categories, or trends with no specific name), return an empty string.
"""


@dataclass
class ResolvedEntity:
    mentioned_as: str
    resolved_name: str | None  # the real, confirmed database value, or None if no good match
    table: str | None  # 'planets' or 'host_stars', whichever table it was found in
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
    """
    Real, deterministic fuzzy match against both pl_name and hostname
    using pg_trgm similarity -- no LLM involved in this step.
    """
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
    """
    Returns a ResolvedEntity if the question mentions a specific named
    planet/star, or None if it doesn't mention one at all (most
    questions -- counts, categories, trends -- have nothing to resolve).
    """
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
        "Tell me about a planet called Zorblax Prime",  # should resolve to nothing / weak match
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
