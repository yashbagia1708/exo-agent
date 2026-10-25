"""
Central configuration for exo-agent.

Defaults to Ollama (free, local) so the project runs at zero cost out of
the box. Switch providers with a single environment variable -- no code
changes anywhere else in the agent.

Usage:
    PROVIDER=ollama python3 run_agent.py "..."          # default, free
    PROVIDER=anthropic python3 run_agent.py "..."        # needs ANTHROPIC_API_KEY
    PROVIDER=openai python3 run_agent.py "..."            # needs OPENAI_API_KEY
"""

import os

from dotenv import load_dotenv

load_dotenv()  # reads a local .env file if present, without overriding real env vars

# --- Provider selection -----------------------------------------------------

PROVIDER = os.getenv("PROVIDER", "ollama").lower()

# Sensible default model per provider. Override with MODEL env var if you
# want a different one (e.g. a smaller/larger Ollama model, or
# claude-haiku-4-5 instead of claude-sonnet-4-6 for cheaper test runs).
#
# qwen2.5-coder:14b is the Ollama default: a code-specialized model
# outperforms general-purpose models of similar size on SQL generation
# and structured-output tasks.
DEFAULT_MODELS = {
    "ollama": "qwen2.5-coder:14b",
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
}

MODEL = os.getenv("MODEL", DEFAULT_MODELS.get(PROVIDER, ""))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# --- Database ----------------------------------------------------------------

# Generic placeholder default -- set the real DB_DSN in your own .env
# (never commit real credentials), pointed at the single database that
# has planets, host_stars, and wiki_chunks all together.
DB_DSN = os.getenv(
    "DB_DSN",
    "dbname=exoagent user=exoagent password=exoagent host=localhost port=5432",
)

# --- Agent behavior ------------------------------------------------------------

MAX_SQL_REPAIR_ATTEMPTS = int(os.getenv("MAX_SQL_REPAIR_ATTEMPTS", "2"))

# --- Validation ---------------------------------------------------------------

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
    """Human-readable summary of the active config, useful for logging/demos."""
    cost_note = "free, local" if PROVIDER == "ollama" else "paid API"
    return f"provider={PROVIDER} model={MODEL} ({cost_note})"
