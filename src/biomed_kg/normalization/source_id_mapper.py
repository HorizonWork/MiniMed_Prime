"""Pseudocode source identifier mapper."""
from __future__ import annotations


class SourceIDMapper:
    def __init__(self, entity_repo):
        self.entity_repo = entity_repo

    def canonical_entity_id(self, source_system: str, source_id: str) -> str | None:
        candidates = self.entity_repo.find_by_identifier(source_system, source_id)
        return candidates[0].entity_id if candidates else None
