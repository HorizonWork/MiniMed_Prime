"""Postgres-backed synonym fallback linker.

Runs when scispacy's UMLS linker can't attach a CUI (or attaches one below
threshold). Performs an exact normalized-text match against the ``term``
table, which has been populated from MRCONSO during UMLS ingest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from minimed_rag.common.ids import make_concept_id
from minimed_rag.nlp.linking.entity_linker import LinkedMention


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


@dataclass(slots=True)
class SynonymFallbackLinker:
    """Exact normalized-text lookup against the Postgres ``term`` table."""

    term_repo: Any
    confidence: float = 0.7
    max_candidates: int = 10

    def link_mention(self, mention_text: str) -> LinkedMention | None:
        normalized = normalize_text(mention_text)
        if not normalized:
            return None
        matches = self.term_repo.search_by_normalized_text(normalized, limit=self.max_candidates)
        if not matches:
            return None
        best = matches[0]
        candidates = [m.cui for m in matches if getattr(m, "cui", None)]
        return LinkedMention(
            mention_text=mention_text,
            expanded_text=mention_text,
            entity_id="",
            concept_id=best.concept_id or make_concept_id(best.cui),
            cui=best.cui,
            semantic_type=None,
            confidence=self.confidence,
            candidates=candidates,
        )
