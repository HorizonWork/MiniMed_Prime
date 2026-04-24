"""Pseudocode concept embedder."""
from __future__ import annotations


class ConceptEmbedder:
    model_name = "concept_embedder"

    def encode_concepts(self, concepts: list) -> list[dict]:
        texts = [f"{concept.preferred_name} {' '.join(concept.semantic_types)}" for concept in concepts]
        vectors = self.encode(texts)
        return [{"concept_id": concept.concept_id, "vector": vector} for concept, vector in zip(concepts, vectors)]

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]
