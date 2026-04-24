"""Pseudocode SapBERT candidate ranker."""
from __future__ import annotations


class SapBERTRanker:
    def __init__(self, encoder):
        self.encoder = encoder

    def __call__(self, mention: str, context: str, candidates: list) -> list:
        mention_vector = self.encoder.encode([mention + " " + context])[0]
        for candidate in candidates:
            candidate.score = self.encoder.cosine(mention_vector, candidate.vector)
        return sorted(candidates, key=lambda item: item.score, reverse=True)
