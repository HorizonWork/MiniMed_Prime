"""Pseudocode embedding model interface."""

from __future__ import annotations


class Embedder:
    model_name: str = "embedder"

    def encode(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError
