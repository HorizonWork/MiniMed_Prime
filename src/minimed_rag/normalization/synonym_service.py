"""Pseudocode synonym service."""

from __future__ import annotations

from minimed_rag.normalization.terminology_service import normalize_text


class SynonymService:
    def __init__(self, term_repo):
        self.term_repo = term_repo

    def expand(self, concept_id: str) -> list[str]:
        return [term.text for term in self.term_repo.get_terms_for_concept(concept_id)]

    def normalize_synonym(self, text: str) -> str:
        return normalize_text(text)
