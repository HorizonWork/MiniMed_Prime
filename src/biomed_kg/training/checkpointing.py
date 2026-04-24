"""Pseudocode checkpointing."""
from __future__ import annotations


class Checkpointing:
    def __init__(self):
        self.best_score = float("-inf")

    def save_if_best(self, model, metrics) -> None:
        score = metrics.auc
        if score > self.best_score:
            self.best_score = score
            self.save(model, metrics)

    def save(self, model, metrics) -> None:
        return None
