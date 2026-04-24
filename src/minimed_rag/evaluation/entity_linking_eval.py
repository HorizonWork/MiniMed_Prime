"""Pseudocode EntityLinkingEvaluator."""

from __future__ import annotations


class EntityLinkingEvaluator:
    def evaluate(self, graph_version: str) -> dict:
        return {"top1_accuracy": 0.0, "top5_accuracy": 0.0, "ambiguous_mention_rate": 0.0}
