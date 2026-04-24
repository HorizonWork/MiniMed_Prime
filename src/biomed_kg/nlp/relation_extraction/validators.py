"""Pseudocode relation extraction validators."""
from __future__ import annotations


class RelationExtractionValidator:
    def __init__(self, predicate_registry):
        self.predicate_registry = predicate_registry

    def keep(self, extracted_relation) -> bool:
        return extracted_relation.get("confidence", 0.0) >= 0.5
