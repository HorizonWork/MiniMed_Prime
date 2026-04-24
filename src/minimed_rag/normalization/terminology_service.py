"""Pseudocode UMLS-centric terminology service."""

from __future__ import annotations


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def intersects(left: list[str], right: list[str]) -> bool:
    return bool(set(left) & set(right))


def rank_term_candidates(text: str, candidates: list):
    normalized = normalize_text(text)
    return sorted(
        candidates, key=lambda candidate: getattr(candidate, "normalized_text", "") != normalized
    )


class TerminologyService:
    def __init__(self, concept_repo, term_repo, identifier_repo):
        self.concept_repo = concept_repo
        self.term_repo = term_repo
        self.identifier_repo = identifier_repo

    def lookup_by_cui(self, cui: str):
        return self.concept_repo.get_by_cui(cui)

    def lookup_by_source_code(self, namespace: str, code: str) -> list:
        return self.identifier_repo.find_concepts(namespace, code)

    def lookup_by_term(self, text: str, semantic_types: list[str] | None = None):
        candidates = self.term_repo.search(normalize_text(text))
        if semantic_types:
            candidates = [
                candidate
                for candidate in candidates
                if intersects(getattr(candidate, "semantic_types", []), semantic_types)
            ]
        return rank_term_candidates(text, candidates)

    def get_synonyms(self, concept_id: str):
        return self.term_repo.get_terms_for_concept(concept_id)

    def get_parents(self, concept_id: str):
        return self.concept_repo.get_parents(concept_id)

    def get_children(self, concept_id: str):
        return self.concept_repo.get_children(concept_id)
