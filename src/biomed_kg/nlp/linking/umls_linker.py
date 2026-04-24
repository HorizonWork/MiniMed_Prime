"""Pseudocode direct UMLS linker."""
from __future__ import annotations


class UMLSLinker:
    def __init__(self, terminology_service):
        self.terminology_service = terminology_service

    def link_cui(self, cui: str):
        return self.terminology_service.lookup_by_cui(cui)

    def link_source_code(self, namespace: str, code: str):
        return self.terminology_service.lookup_by_source_code(namespace, code)
