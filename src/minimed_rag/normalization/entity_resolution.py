"""Pseudocode entity resolution pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MergeDecision:
    action: str
    target_entity_id: str | None
    confidence: float


class EntityResolutionPipeline:
    def __init__(self, lakehouse, entity_repo, terminology_service):
        self.lakehouse = lakehouse
        self.entity_repo = entity_repo
        self.terminology_service = terminology_service

    def run(self) -> None:
        source_entities = self.lakehouse.read("normalized_source_entity")
        for entity in source_entities:
            candidates = self.find_merge_candidates(entity)
            decision = self.decide_merge(entity, candidates)
            if decision.action == "merge":
                self.merge(entity, decision.target_entity_id)
            elif decision.action == "create":
                self.create_canonical_entity(entity)
            elif decision.action == "review":
                self.enqueue_review(entity, candidates)

    def find_merge_candidates(self, entity) -> list:
        candidates = []
        candidates += self.entity_repo.find_by_identifier(entity.namespace, entity.source_id)
        if getattr(entity, "cui", None):
            candidates += self.entity_repo.find_by_cui(entity.cui)
        for xref in getattr(entity, "xrefs", []):
            candidates += self.entity_repo.find_by_identifier(xref.namespace, xref.identifier)
        candidates += self.terminology_service.lookup_by_term(
            entity.name, semantic_types=getattr(entity, "semantic_types", None)
        )
        return list(
            {
                getattr(candidate, "entity_id", str(candidate)): candidate
                for candidate in candidates
            }.values()
        )

    def decide_merge(self, entity, candidates) -> MergeDecision:
        for candidate in candidates:
            if self.exact_identifier_match(entity, candidate):
                return MergeDecision("merge", candidate.entity_id, 0.99)
            if self.same_cui_and_compatible_type(entity, candidate):
                return MergeDecision("merge", candidate.entity_id, 0.97)
        if candidates:
            return MergeDecision("review", None, 0.50)
        return MergeDecision("create", None, 1.0)

    def exact_identifier_match(self, entity, candidate) -> bool:
        return getattr(entity, "source_id", None) in sum(
            getattr(candidate, "identifiers", {}).values(), []
        )

    def same_cui_and_compatible_type(self, entity, candidate) -> bool:
        return bool(
            getattr(entity, "cui", None) and entity.cui == getattr(candidate, "canonical_cui", None)
        )

    def merge(self, entity, target_entity_id: str | None) -> None:
        self.entity_repo.merge(entity, target_entity_id)

    def create_canonical_entity(self, entity) -> None:
        self.entity_repo.create(entity)

    def enqueue_review(self, entity, candidates) -> None:
        self.entity_repo.enqueue_review(entity, candidates)
