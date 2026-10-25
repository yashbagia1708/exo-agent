"""
Returns a single, uniform client for whichever provider is configured.

Every LangGraph node calls this the same way, regardless of provider:

    from exo_agent.providers.factory import get_client
    from exo_agent import config

    client = get_client()
    result = client.chat.completions.create(
        model=config.MODEL,
        response_model=SomePydanticModel,
        messages=[{"role": "user", "content": "..."}],
    )

`result` is always a validated instance of SomePydanticModel, whether it
came from Ollama (local Llama) or Anthropic (Claude). The node code never
branches on provider -- that branching lives entirely here, in one place.
"""

import instructor
from anthropic import Anthropic
from openai import OpenAI

from exo_agent import config


def get_client():
    if config.PROVIDER == "anthropic":
        return instructor.from_anthropic(Anthropic())

    if config.PROVIDER == "openai":
        return instructor.from_openai(OpenAI())

    if config.PROVIDER == "ollama":
        # Ollama exposes an OpenAI-compatible endpoint, so we reuse the
        # OpenAI client pointed at localhost instead of a real API key.
        return instructor.from_openai(
            OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama"),
            mode=instructor.Mode.JSON,
        )

    raise ValueError(f"No client wiring for provider '{config.PROVIDER}'")
