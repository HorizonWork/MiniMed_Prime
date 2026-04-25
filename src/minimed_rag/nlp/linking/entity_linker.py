"""Entity linking.

Two offerings in this module:

- ``EntityLinker`` (Phase 5+ orchestrator): the full terminology-service-
  powered pipeline (abbreviations → candidates → SapBERT rank). It depends
  on ``term_repo``/``concept_repo``/``identifier_repo``/``sapbert_ranker``
  which are planned work; leave it as scaffolding.
- ``QuestionEntityLinker`` (Phase 4 MVP): a simpler 3-tier pipeline wired
  against scispacy and the Postgres-backed synonym fallback. This is what
  ``minimed rag query --use-kg`` uses today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class LinkedMention:
    mention_text: str
    expanded_text: str
    entity_id: str
    concept_id: str | None
    cui: str | None
    semantic_type: str | None
    confidence: float
    candidates: list = field(default_factory=list)


class EntityLinker:
    """Phase 5+ orchestrator — terminology-service + SapBERT rank."""

    def __init__(
        self, terminology_service, abbreviation_resolver, candidate_generator, sapbert_ranker
    ):
        self.terminology = terminology_service
        self.abbreviation_resolver = abbreviation_resolver
        self.candidate_generator = candidate_generator
        self.rank = sapbert_ranker

    def link_mentions(self, text: str, mentions: list) -> list[LinkedMention]:
        linked_mentions = []
        abbreviations = self.abbreviation_resolver.resolve(text)
        for mention in mentions:
            surface = mention.text
            expanded_surface = abbreviations.get(surface, surface)
            candidates = self.candidate_generator.generate(
                text=expanded_surface, semantic_hint=mention.entity_type, context=text
            )
            ranked = self.rank(mention=expanded_surface, context=text, candidates=candidates)
            if not ranked:
                continue
            best = ranked[0]
            linked_mentions.append(
                LinkedMention(
                    surface,
                    expanded_surface,
                    best.entity_id,
                    getattr(best, "concept_id", None),
                    getattr(best, "cui", None),
                    getattr(best, "semantic_type", None),
                    best.score,
                    ranked[:10],
                )
            )
        return linked_mentions

    def link_answer(self, answer) -> list:
        mentions = [type("Mention", (), {"text": answer.text, "entity_type": None})()]
        return self.link_mentions(answer.text, mentions)


@dataclass
class QuestionEntityLinker:
    """Phase 4 MVP 3-tier linker.

    Tier 1 — scispacy NER + UMLS linker (``ScispacyEntityLinker``).
    Tier 2 — Postgres ``term`` table exact-normalized-text fallback
             (``SynonymFallbackLinker``) when scispacy misses or falls
             below ``min_confidence``.
    Tier 3 — Neo4j entity resolution: once a CUI is attached, query
             ``GraphClient.find_entity_by_cui`` to materialize the
             ``Entity.entity_id`` (linking PrimeKG rows, not just UMLS).

    Designed to tolerate missing components — pass ``None`` for the
    fallback or graph_client if you want NER-only behavior.
    """

    scispacy_linker: Any
    fallback_linker: Any = None
    graph_client: Any = None
    min_confidence: float = 0.5

    def link_question(self, text: str) -> list[LinkedMention]:
        raw_mentions: list[LinkedMention] = self.scispacy_linker.link(text)
        resolved: list[LinkedMention] = []
        for mention in raw_mentions:
            final = self._resolve_mention(mention)
            if final is None:
                continue
            resolved.append(final)
        return resolved

    def _resolve_mention(self, mention: LinkedMention) -> LinkedMention | None:
        if mention.cui and mention.confidence >= self.min_confidence:
            return self._attach_entity_id(mention)

        if self.fallback_linker is not None:
            fallback = self.fallback_linker.link_mention(mention.mention_text)
            if fallback is not None and fallback.cui:
                # Preserve scispacy's semantic_type when the fallback doesn't have one.
                if not fallback.semantic_type and mention.semantic_type:
                    fallback.semantic_type = mention.semantic_type
                return self._attach_entity_id(fallback)

        # Tier 3 drops to NER-only: retain mention if it has a semantic type.
        if mention.semantic_type:
            return mention
        return None

    def _attach_entity_id(self, mention: LinkedMention) -> LinkedMention:
        if self.graph_client is None or not mention.cui:
            return mention
        matches = self.graph_client.find_entity_by_cui(mention.cui, limit=1)
        if matches:
            mention.entity_id = matches[0].entity_id or mention.entity_id
        return mention
