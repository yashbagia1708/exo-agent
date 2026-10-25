# exo-agent — live demo frontend

A single-question UI on top of the real agent in `src/exo_agent/graph.py`.
No mocked responses — every answer comes from an actual run of the agent
against the live Postgres database.

## Run it

Make sure these are already running/loaded from the main setup:
- Postgres with the data loaded (`data/load_to_postgres.py`)
- Ollama running (`ollama serve`, in its own terminal) with a pulled model

Then, from the project root:

```bash
pip install -r requirements.txt
pip install -e .

PROVIDER=ollama uvicorn frontend.server:app --reload --port 8420
```

Open **http://localhost:8420** in your browser.

To demo with Claude instead of the local model (costs a small amount per
question — see the main README):

```bash
PROVIDER=anthropic uvicorn frontend.server:app --reload --port 8420
```

## What's actually happening

The frontend calls one endpoint, `POST /api/ask`, which calls `ask()` in
`src/exo_agent/graph.py` directly — the exact same LangGraph agent used
by the CLI and the eval harness. The "How it got this answer" panel is
built from the real intermediate state of that graph run (the actual SQL
written, the actual row count returned, the actual chart decision) — it's
not a separate, simplified explanation generated after the fact.
