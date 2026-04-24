"""Pseudocode source predicate mapper."""
from __future__ import annotations


class PredicateMapper:
    def __init__(self, predicate_registry):
        self.registry = predicate_registry

    def map_source_predicate(self, source_system: str, source_predicate: str, subject_type: str, object_type: str) -> str:
        mappings = self.registry.get_source_mappings(source_system, source_predicate)
        valid_mappings = []
        for candidate_predicate in mappings:
            if candidate_predicate in self.registry.all() and self.registry.is_valid_domain_range(candidate_predicate, subject_type, object_type):
                valid_mappings.append(candidate_predicate)
        if len(valid_mappings) == 1:
            return valid_mappings[0]
        if len(valid_mappings) > 1:
            return self.choose_most_specific_predicate(valid_mappings)
        return "associated_with"

    def choose_most_specific_predicate(self, predicates: list[str]) -> str:
        return sorted(predicates)[0]
