"""Pseudocode domain object: canonical entity."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Entity:
    entity_id: str
    entity_type: str
    preferred_name: str
    canonical_cui: str | None = None
    identifiers: dict[str, list[str]] = field(default_factory=dict)
    semantic_types: list[str] = field(default_factory=list)
    is_active: bool = True
    graph_version: str = "kg_local"
