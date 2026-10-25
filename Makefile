# exo-agent -- common tasks. Run `make help` for the list.

PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin
PORT   ?= 8420

.DEFAULT_GOAL := help
.PHONY: help setup db-up db-down db-reset download-data load-data load-wiki \
        models serve test lint format eval clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: $(BIN)/activate .env  ## Create .venv and install the project + dev tools

$(BIN)/activate: pyproject.toml
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"
	@touch $(BIN)/activate

.env:
	cp .env.example .env
	@echo "Created .env from .env.example -- edit it if your setup differs."

db-up:  ## Start Postgres (pgvector + pg_trgm) in Docker and wait until ready
	docker compose up -d --wait db

db-down:  ## Stop Postgres (data is kept in the Docker volume)
	docker compose down

db-reset:  ## Stop Postgres and DELETE its data volume
	docker compose down -v

download-data:  ## Fetch the NASA Exoplanet Archive table (needs internet)
	$(BIN)/python data/download_data.py

models:  ## Pull the Ollama chat + embedding models
	ollama pull qwen2.5-coder:14b
	ollama pull nomic-embed-text

load-data: data/raw_pscomppars.csv  ## Load planets/host_stars, then the wiki corpus
	$(BIN)/python data/load_to_postgres.py
	$(MAKE) load-wiki

load-wiki:  ## Embed and load the wiki corpus (needs Ollama running)
	$(BIN)/python data/load_wiki_to_postgres.py

data/raw_pscomppars.csv:
	$(MAKE) download-data

serve:  ## Run the demo UI (http://localhost:8420, override with PORT=)
	$(BIN)/uvicorn frontend.server:app --reload --port $(PORT)

test:  ## Run the deterministic test suite (no LLM or database needed)
	$(BIN)/pytest -q

lint:  ## Check style and imports
	$(BIN)/ruff check .

format:  ## Auto-fix lint issues and format
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

eval:  ## Run the LLM eval harness (slow; needs Ollama + loaded DB)
	$(BIN)/python eval/run_eval.py

clean:  ## Remove caches and generated charts
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -f charts/*.png
