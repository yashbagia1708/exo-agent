"""
Checks against a real, loaded database. Skipped unless REQUIRE_DB=1:

    make db-up load-data
    REQUIRE_DB=1 pytest -q tests/integration      (or: make test-db)

The pipeline workflow runs these on GitHub after loading NASA's live table.
Needs Ollama for the search test (embeddings); no chat model.
"""

import os

import psycopg2
import pytest

from exo_agent import config

pytestmark = pytest.mark.skipif(os.getenv("REQUIRE_DB") != "1", reason="needs a loaded database (REQUIRE_DB=1)")


def _one(sql, params=None):
    with psycopg2.connect(config.DB_DSN) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def test_tables_are_loaded_and_consistent():
    planets, stars, chunks = _one(
        "SELECT (SELECT count(*) FROM planets), (SELECT count(*) FROM host_stars), (SELECT count(*) FROM wiki_chunks)"
    )
    assert planets > 5000 and stars > 4000 and chunks > 400
    # every planet's star exists, and every star has at least one planet
    assert _one("SELECT count(*) FROM planets p LEFT JOIN host_stars h USING (hostname) WHERE h.hostname IS NULL")[0] == 0
    assert _one("SELECT count(*) FROM host_stars h WHERE NOT EXISTS (SELECT 1 FROM planets p WHERE p.hostname = h.hostname)")[0] == 0


def test_extensions_and_search_columns():
    names = _one("SELECT array_agg(extname::text) FROM pg_extension")[0]
    assert {"vector", "pg_trgm"} <= set(names)
    # every chunk has both halves of hybrid search
    assert _one("SELECT count(*) FROM wiki_chunks WHERE embedding IS NULL OR search_vec IS NULL")[0] == 0


def test_hybrid_search_finds_the_right_article():
    from exo_agent.retrieval import hybrid_search

    top = hybrid_search("What is the Kepler space telescope?", top_k=3)
    assert top[0].title == "Kepler space telescope"
    assert top[0].lexical_rank and top[0].semantic_rank  # found by both searches


def test_entity_matching_uses_the_stored_name():
    from exo_agent.nodes.entity_resolver import _trigram_lookup

    hit = _trigram_lookup("Proxima Centauri b")
    assert hit.resolved_name == "Proxima Cen b" and hit.table == "planets"
    assert _trigram_lookup("Zorblax Prime").resolved_name is None


def test_zero_row_check_finds_tess_in_the_right_column():
    from exo_agent.nodes.executor import empty_result_hint

    hint = empty_result_hint("SELECT pl_name FROM planets WHERE discoverymethod ILIKE '%TESS%'")
    assert "disc_facility" in hint and "Transiting Exoplanet Survey Satellite" in hint
    assert empty_result_hint("SELECT pl_name FROM planets WHERE hostname ILIKE '%Zorblax%'") is None


def test_largest_planet_is_never_a_missing_radius():
    from exo_agent.nodes.executor import execute_sql

    # no repair expected, so no model is called
    result = execute_sql(
        "SELECT pl_name, pl_rade FROM planets WHERE disc_facility ILIKE '%TESS%' ORDER BY pl_rade DESC LIMIT 1",
        "largest planet found by TESS",
    )
    assert result.success and result.repair_attempts == 0
    assert result.rows[0]["pl_rade"] is not None
    assert "NULLS LAST" in result.sql_used
