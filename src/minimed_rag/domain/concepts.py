"""Pseudocode domain object: terminology concept."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Concept:
    concept_id: str
    cui: str | None
    preferred_name: str
    semantic_types: list[str] = field(default_factory=list)
    parent_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    graph_version: str = "kg_local"
