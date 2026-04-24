"""Pseudocode entity-linking retriever."""

from __future__ import annotations


class EntityLinkingRetriever:
    def __init__(self, entity_linker, ner):
        self.entity_linker = entity_linker
        self.ner = ner

    def retrieve(self, query: str) -> list:
        return self.entity_linker.link_mentions(query, self.ner.extract(query))
