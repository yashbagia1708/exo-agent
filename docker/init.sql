-- Runs once, the first time the Postgres volume is created.
-- pgvector backs semantic search (wiki_chunks.embedding);
-- pg_trgm backs fuzzy entity resolution (nodes/entity_resolver.py).
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
