"""Schema-driven validation for runtime KG assertions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from minimed_rag.schema_registry.predicate_registry import PredicateRegistry


@dataclass(frozen=True, slots=True)
class AssertionDirectionValidator:
    """Validate assertion direction using predicate domain/range schema.

    Unknown predicates and missing endpoint types are allowed so retrieval
    remains robust while the KG schema is still growing. Known predicates with
    typed endpoints must match the registry domain/range.
    """

    predicate_registry: Any | None = None
    allow_unknown_predicates: bool = True
    allow_missing_types: bool = True

    def __post_init__(self) -> None:
        if self.predicate_registry is None:
            object.__setattr__(self, "predicate_registry", PredicateRegistry())

    def is_valid(self, predicate: str, subject_type: str | None, object_type: str | None) -> bool:
        predicate = (predicate or "").strip()
        if not predicate:
            return self.allow_unknown_predicates
        if not self._has_rule(predicate):
            return self.allow_unknown_predicates
        if not subject_type or not object_type:
            return self.allow_missing_types
        return bool(
            self.predicate_registry.is_valid_domain_range(predicate, subject_type, object_type)
        )

    def _has_rule(self, predicate: str) -> bool:
        try:
            rules = self.predicate_registry.all()
        except AttributeError:
            rules = getattr(self.predicate_registry, "rules", {})
        return predicate in rules
