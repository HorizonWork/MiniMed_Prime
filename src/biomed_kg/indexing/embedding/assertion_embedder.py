"""Pseudocode assertion embedder."""
from __future__ import annotations


class AssertionEmbedder:
    model_name = "assertion_embedder"

    def prepare_assertion_text(self, assertion, entity_repo) -> str:
        subject = entity_repo.get(assertion.subject_id)
        obj = entity_repo.get(assertion.object_id)
        return f"{subject.preferred_name} {assertion.predicate} {obj.preferred_name}"

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]
