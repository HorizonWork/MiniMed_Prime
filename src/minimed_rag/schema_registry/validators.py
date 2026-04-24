"""Pseudocode validators for registry-backed rules."""

from __future__ import annotations

from minimed_rag.common.exceptions import InvalidAssertionError


class AssertionValidator:
    def __init__(self, predicate_registry):
        self.predicate_registry = predicate_registry

    def validate_domain_range(self, predicate: str, subject_type: str, object_type: str) -> None:
        if not self.predicate_registry.is_valid_domain_range(predicate, subject_type, object_type):
            raise InvalidAssertionError(predicate, subject_type, object_type)
