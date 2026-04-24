"""Pseudocode repository for canonical entities."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EntityRepo:
    rows: dict[str, object] = field(default_factory=dict)
    identifier_index: dict[tuple[str, str], list[object]] = field(default_factory=dict)
    cui_index: dict[str, list[object]] = field(default_factory=dict)
    source_index: dict[str, object] = field(default_factory=dict)

    def get(self, entity_id: str):
        return self.rows.get(entity_id)

    def upsert(self, entity) -> None:
        self.rows[entity.entity_id] = entity
        if entity.canonical_cui:
            self.cui_index.setdefault(entity.canonical_cui, []).append(entity)
        for namespace, identifiers in entity.identifiers.items():
            for identifier in identifiers:
                self.identifier_index.setdefault((namespace, identifier), []).append(entity)

    def find_by_identifier(self, namespace: str, identifier: str) -> list:
        return self.identifier_index.get((namespace, identifier), [])

    def find_by_cui(self, cui: str) -> list:
        return self.cui_index.get(cui, [])

    def get_canonical_by_source_id(self, source_id: str):
        return self.source_index[source_id]

    def get_type(self, entity_id: str) -> str:
        return self.rows[entity_id].entity_type

    def merge(self, source_entity, target_entity_id: str | None) -> None:
        if target_entity_id:
            self.source_index[source_entity.source_id] = self.rows[target_entity_id]

    def create(self, entity) -> None:
        self.rows[entity.entity_id] = entity

    def enqueue_review(self, entity, candidates) -> None:
        return None
