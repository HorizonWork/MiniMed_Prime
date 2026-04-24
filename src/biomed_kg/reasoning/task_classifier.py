"""Pseudocode biomedical reasoning task classifier."""
from __future__ import annotations


class TaskClassifier:
    def classify(self, query: str) -> str:
        lowered = query.casefold()
        if "adverse" in lowered or "cause" in lowered or "toxicity" in lowered:
            return "adverse_effect"
        if "treat" in lowered or "therapy" in lowered:
            return "treatment"
        if "mechanism" in lowered or "pathway" in lowered:
            return "mechanism"
        return "rag"
