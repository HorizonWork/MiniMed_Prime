"""Pseudocode entity type registry."""

from __future__ import annotations

from minimed_rag.common.config import load_yaml


class EntityTypeRegistry:
    def __init__(self, path: str = "configs/schema/entity_types.yaml"):
        self.types = load_yaml(path).get("entity_types", {})

    def exists(self, entity_type: str) -> bool:
        return entity_type in self.types

    def is_child_of(self, entity_type: str, parent_type: str) -> bool:
        current = entity_type
        while current in self.types:
            if current == parent_type:
                return True
            current = self.types.get(current, {}).get("parent")
        return parent_type == "Entity"
