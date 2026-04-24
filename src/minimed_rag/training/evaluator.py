"""Pseudocode TRM evaluator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TRMMetrics:
    auc: float
    f1: float
    path_accuracy: float


class TRMEvaluator:
    def evaluate(self, model, data_loader) -> TRMMetrics:
        return TRMMetrics(auc=0.0, f1=0.0, path_accuracy=0.0)
