"""Pseudocode reranker."""

from __future__ import annotations


class Reranker:
    def __init__(self, model=None):
        self.model = model

    def rerank(self, query: str, results: list) -> list:
        for result in results:
            if self.model:
                result.score = self.model.score(query, result)
        return sorted(results, key=lambda item: item.score, reverse=True)
