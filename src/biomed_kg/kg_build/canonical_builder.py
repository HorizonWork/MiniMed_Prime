"""Pseudocode canonical KG builder."""
from __future__ import annotations


class CanonicalBuilder:
    def __init__(self, lakehouse, assertion_builder, assertion_repo):
        self.lakehouse = lakehouse
        self.assertion_builder = assertion_builder
        self.assertion_repo = assertion_repo

    def build_assertions(self) -> None:
        for source_assertion in self.lakehouse.read("normalized_source_assertion"):
            assertion = self.assertion_builder.build_from_source_assertion(source_assertion)
            self.assertion_repo.upsert(assertion)
