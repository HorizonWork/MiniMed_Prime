"""Pseudocode ontology alignment."""
from __future__ import annotations


class OntologyAlignment:
    def __init__(self, semantic_type_registry):
        self.semantic_type_registry = semantic_type_registry

    def infer_entity_type(self, semantic_types: list[str], source_type: str | None = None) -> str:
        for semantic_type in semantic_types:
            entity_type = self.semantic_type_registry.entity_type_for(semantic_type)
            if entity_type:
                return entity_type
        return source_type or "Entity"

    def align_concept(self, source_concept) -> dict:
        return {"concept_id": source_concept.concept_id, "entity_type": self.infer_entity_type(source_concept.semantic_types)}
