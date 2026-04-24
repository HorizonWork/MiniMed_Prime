"""Pseudocode query understanding."""
from __future__ import annotations


class QueryUnderstanding:
    def __init__(self, ner, entity_linker, task_classifier):
        self.ner = ner
        self.entity_linker = entity_linker
        self.task_classifier = task_classifier

    def analyze(self, query: str) -> dict:
        mentions = self.ner.extract(query)
        linked_entities = self.entity_linker.link_mentions(query, mentions)
        task_type = self.task_classifier.classify(query)
        return {"query": query, "mentions": mentions, "linked_entities": linked_entities, "task_type": task_type}
