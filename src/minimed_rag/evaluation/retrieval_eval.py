"""Pseudocode RetrievalEvaluator."""

from __future__ import annotations


class RetrievalEvaluator:
    def evaluate(self, graph_version: str) -> dict:
        return {"recall@k": 0.0, "evidence_recall@k": 0.0, "mrr": 0.0, "nDCG": 0.0}
