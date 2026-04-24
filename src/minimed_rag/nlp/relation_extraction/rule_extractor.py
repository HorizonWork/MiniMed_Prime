"""Pseudocode rule-based relation extraction."""

from __future__ import annotations


class RuleExtractor:
    def __init__(self, rules):
        self.rules = rules

    def extract(self, chunk, linked_mentions: list) -> list:
        assertions = []
        for rule in self.rules:
            if rule.matches(chunk.text, linked_mentions):
                assertions.append(rule.to_source_assertion(chunk, linked_mentions))
        return assertions
