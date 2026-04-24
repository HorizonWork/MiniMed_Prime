"""Pseudocode PubTator client."""

from __future__ import annotations

from minimed_rag.nlp.ner.base import Mention


class PubTatorClient:
    def __init__(self, http_client):
        self.http_client = http_client

    def extract(self, text: str) -> list[Mention]:
        response = self.http_client.annotate(text)
        return [
            Mention(
                item["text"], item["start"], item["end"], item.get("type"), item.get("score", 1.0)
            )
            for item in response
        ]
