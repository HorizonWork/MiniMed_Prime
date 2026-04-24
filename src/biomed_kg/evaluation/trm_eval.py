"""Pseudocode TRMEvaluator."""
from __future__ import annotations


class TRMEvaluator:
    def evaluate_latest(self, graph_version: str) -> dict:
        return {"path_validity_auc": 0.0, "answer_selection_accuracy": 0.0, "hard_negative_accuracy": 0.0, "task_type_breakdown": {}}
