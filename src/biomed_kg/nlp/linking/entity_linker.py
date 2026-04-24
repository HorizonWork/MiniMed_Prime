"""Pseudocode entity linker orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field


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
    def __init__(self, terminology_service, abbreviation_resolver, candidate_generator, sapbert_ranker):
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
            candidates = self.candidate_generator.generate(text=expanded_surface, semantic_hint=mention.entity_type, context=text)
            ranked = self.rank(mention=expanded_surface, context=text, candidates=candidates)
            if not ranked:
                continue
            best = ranked[0]
            linked_mentions.append(LinkedMention(surface, expanded_surface, best.entity_id, getattr(best, "concept_id", None), getattr(best, "cui", None), getattr(best, "semantic_type", None), best.score, ranked[:10]))
        return linked_mentions

    def link_answer(self, answer) -> list:
        mentions = [type("Mention", (), {"text": answer.text, "entity_type": None})()]
        return self.link_mentions(answer.text, mentions)
