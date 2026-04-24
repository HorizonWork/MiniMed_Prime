"""Pseudocode model-based relation extraction."""
from __future__ import annotations


class ModelExtractor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def extract(self, chunk, linked_mentions: list) -> list:
        pairs = self.enumerate_mention_pairs(linked_mentions)
        return [self.predict_relation(chunk, pair) for pair in pairs]

    def enumerate_mention_pairs(self, mentions: list) -> list[tuple]:
        return [(a, b) for i, a in enumerate(mentions) for b in mentions[i + 1:]]

    def predict_relation(self, chunk, pair):
        return {"chunk_id": chunk.chunk_id, "subject": pair[0], "object": pair[1], "predicate": "associated_with", "confidence": 0.0}
