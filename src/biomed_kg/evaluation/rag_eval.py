"""Pseudocode RAGEvaluator."""
from __future__ import annotations


class RAGEvaluator:
    def evaluate(self, graph_version: str) -> dict:
        return {"groundedness": 0.0, "citation_precision": 0.0, "answer_helpfulness": 0.0}
