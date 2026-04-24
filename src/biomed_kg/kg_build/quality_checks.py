"""Pseudocode canonical KG quality checks."""
from __future__ import annotations


class KGQualityChecks:
    def __init__(self, predicate_registry):
        self.predicate_registry = predicate_registry

    def validate_domain_range(self, assertions: list, entity_repo) -> list[str]:
        errors = []
        for assertion in assertions:
            subject_type = entity_repo.get_type(assertion.subject_id)
            object_type = entity_repo.get_type(assertion.object_id)
            if not self.predicate_registry.is_valid_domain_range(assertion.predicate, subject_type, object_type):
                errors.append(assertion.assertion_id)
        return errors
