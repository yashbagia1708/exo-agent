"""
One instructor client for whichever provider is configured.

    client = get_client()
    result = client.chat.completions.create(
        model=config.MODEL,
        response_model=SomePydanticModel,
        messages=[{"role": "user", "content": "..."}],
    )

`result` comes back as a validated SomePydanticModel. Nodes never check
the provider themselves; that only happens here.
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
        # Ollama speaks the OpenAI API; the key is a dummy
        return instructor.from_openai(
            OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama"),
            mode=instructor.Mode.JSON,
        )

    raise ValueError(f"No client wiring for provider '{config.PROVIDER}'")
