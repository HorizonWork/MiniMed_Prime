from __future__ import annotations

import random
import re

from .base import ModelProvider


class RandomBaseline(ModelProvider):
    """Uniformly random answer selection — useful for sanity-checking the harness."""

    name = "random"

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._seed = seed

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        # Detect option letters present in the prompt
        letters = re.findall(r"\b([A-E])\.", prompt)
        seen = list(dict.fromkeys(letters))  # deduplicated, order-preserving
        if seen:
            return self._rng.choice(seen)
        # Fall back to yes/no/maybe for boolean tasks
        if "yes" in prompt.lower() and "no" in prompt.lower():
            return self._rng.choice(["yes", "no", "maybe"])
        return self._rng.choice(["A", "B", "C", "D"])

    @property
    def config(self) -> dict:
        return {"provider": self.name, "seed": self._seed}
