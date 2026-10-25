"""
Smoke test for the provider layer.

Proves two things before we build any real agent logic on top:
  1. instructor + the configured provider can return a validated
     Pydantic object (not just free text)
  2. swapping PROVIDER env var changes who answers, with zero code changes

Run it like:
    PROVIDER=ollama python3 test_provider.py
    PROVIDER=anthropic python3 test_provider.py
"""

import time

from pydantic import BaseModel, Field

from exo_agent import config
from exo_agent.providers.factory import get_client


class FunFact(BaseModel):
    topic: str = Field(description="What this fact is about")
    fact: str = Field(description="One short, true sentence about the topic")
    confidence: float = Field(description="0.0-1.0, how confident you are this fact is accurate")


def main():
    print(f"Testing provider config: {config.describe()}\n")

    client = get_client()

    start = time.time()
    result = client.chat.completions.create(
        model=config.MODEL,
        response_model=FunFact,
        messages=[
            {
                "role": "user",
                "content": "Give me one fun, true fact about exoplanets.",
            }
        ],
    )
    elapsed = time.time() - start

    print("Structured result:")
    print(f"  topic:      {result.topic}")
    print(f"  fact:       {result.fact}")
    print(f"  confidence: {result.confidence}")
    print(f"\nElapsed: {elapsed:.2f}s")
    print("\nSUCCESS — structured output worked end-to-end.")


if __name__ == "__main__":
    main()
