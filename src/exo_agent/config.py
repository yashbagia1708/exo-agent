"""
Settings, read from the environment (.env).

    PROVIDER=ollama      local and free (default)
    PROVIDER=anthropic   needs ANTHROPIC_API_KEY
    PROVIDER=openai      needs OPENAI_API_KEY
"""

import os

from dotenv import load_dotenv

load_dotenv()  # .env never overrides variables already set in the shell

# ------------------------------------------------------------------ provider
PROVIDER = os.getenv("PROVIDER", "ollama").lower()

# Default model per provider; override with MODEL=...
# qwen2.5-coder beat llama3.1:8b on our SQL questions (see FINDINGS.md)
DEFAULT_MODELS = {
    "ollama": "qwen2.5-coder:14b",
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
}

MODEL = os.getenv("MODEL", DEFAULT_MODELS.get(PROVIDER, ""))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# ------------------------------------------------------------------ database
# Default is the Docker Postgres from `make db-up`. One database holds
# planets, host_stars and wiki_chunks.
DB_DSN = os.getenv(
    "DB_DSN",
    "dbname=exoagent user=exoagent password=exoagent host=localhost port=5433",
)

# ------------------------------------------------------------------ agent
MAX_SQL_REPAIR_ATTEMPTS = int(os.getenv("MAX_SQL_REPAIR_ATTEMPTS", "2"))

# ------------------------------------------------------------------ fail fast on bad config
SUPPORTED_PROVIDERS = {"ollama", "anthropic", "openai"}

if PROVIDER not in SUPPORTED_PROVIDERS:
    raise ValueError(
        f"Unknown PROVIDER '{PROVIDER}'. Must be one of: {', '.join(sorted(SUPPORTED_PROVIDERS))}"
    )

if PROVIDER == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
    raise EnvironmentError(
        "PROVIDER=anthropic requires ANTHROPIC_API_KEY to be set (e.g. in a .env file)."
    )

if PROVIDER == "openai" and not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError(
        "PROVIDER=openai requires OPENAI_API_KEY to be set (e.g. in a .env file)."
    )


def describe() -> str:
    """'provider=ollama model=qwen2.5-coder:14b (free, local)'"""
    cost_note = "free, local" if PROVIDER == "ollama" else "paid API"
    return f"provider={PROVIDER} model={MODEL} ({cost_note})"
