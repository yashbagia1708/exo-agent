"""
Row counts for every table, as JSON. The pipeline workflow runs it before
and after re-loading, to prove the loaders can be re-run safely.

    python scripts/db_counts.py
"""

import json

import psycopg2

from exo_agent import config

TABLES = ["planets", "host_stars", "wiki_chunks"]

with psycopg2.connect(config.DB_DSN) as conn, conn.cursor() as cur:
    counts = {}
    for table in TABLES:
        cur.execute(f"SELECT count(*) FROM {table}")
        counts[table] = cur.fetchone()[0]

print(json.dumps(counts, indent=2))
