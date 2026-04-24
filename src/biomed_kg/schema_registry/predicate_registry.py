"""Pseudocode predicate registry loaded from YAML policy."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from biomed_kg.common.config import load_yaml


@dataclass(frozen=True, slots=True)
class PredicateRule:
    name: str
    id: str
    domain: list[str]
    range: list[str]
    directional: bool
    inverse: str
    allowed_tasks: list[str]
    disallowed_tasks: list[str]
    min_confidence_for_training: float
    default_weight: float
    allow_as_gold_path: bool
    max_per_path: int | None = None


class PredicateRegistry:
    def __init__(self, path: str | Path = "configs/schema/predicates.yaml"):
        raw = load_yaml(path).get("predicates", {})
        self.rules = {name: PredicateRule(name=name, **values) for name, values in raw.items()}
        self.source_mappings: dict[tuple[str, str], list[str]] = {}

    def get(self, predicate: str) -> PredicateRule:
        return self.rules[predicate]

    def all(self) -> dict[str, PredicateRule]:
        return self.rules

    def is_directional(self, predicate: str) -> bool:
        return self.rules[predicate].directional

    def is_valid_domain_range(self, predicate: str, subject_type: str, object_type: str) -> bool:
        rule = self.rules[predicate]
        return self._matches(subject_type, rule.domain) and self._matches(object_type, rule.range)

    def get_source_mappings(self, source_system: str, source_predicate: str) -> list[str]:
        return self.source_mappings.get((source_system, source_predicate), [source_predicate])

    def sample_incompatible_predicate(self, subject_type: str, object_type: str, original_predicate: str) -> str:
        for predicate in self.rules:
            if predicate != original_predicate and not self.is_valid_domain_range(predicate, subject_type, object_type):
                return predicate
        return "associated_with"

    @staticmethod
    def _matches(entity_type: str, allowed: list[str]) -> bool:
        return "Entity" in allowed or entity_type in allowed
