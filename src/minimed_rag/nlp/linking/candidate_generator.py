"""Pseudocode entity-linking candidate generator."""

from __future__ import annotations


class CandidateGenerator:
    def __init__(self, terminology_service):
        self.terminology_service = terminology_service

    def generate(
        self, text: str, semantic_hint: str | None = None, context: str | None = None
    ) -> list:
        semantic_types = [semantic_hint] if semantic_hint else None
        return self.terminology_service.lookup_by_term(text, semantic_types=semantic_types)
