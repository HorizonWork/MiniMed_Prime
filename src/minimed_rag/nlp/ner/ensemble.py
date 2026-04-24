"""Pseudocode NER ensemble."""

from __future__ import annotations


class NEREnsemble:
    def __init__(self, models):
        self.models = models

    def extract(self, text: str) -> list:
        mentions = []
        for model in self.models:
            mentions.extend(model.extract(text))
        return self.deduplicate(mentions)

    def deduplicate(self, mentions: list) -> list:
        by_span = {}
        for mention in mentions:
            key = (mention.start, mention.end, mention.text.casefold())
            if key not in by_span or mention.score > by_span[key].score:
                by_span[key] = mention
        return list(by_span.values())
