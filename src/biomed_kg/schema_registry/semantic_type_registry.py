"""Pseudocode semantic type registry."""
from __future__ import annotations

from biomed_kg.common.config import load_yaml


class SemanticTypeRegistry:
    def __init__(self, path: str = "configs/schema/semantic_types.yaml"):
        self.types = load_yaml(path).get("semantic_types", {})

    def entity_type_for(self, semantic_type_id: str) -> str | None:
        return self.types.get(semantic_type_id, {}).get("entity_type")
