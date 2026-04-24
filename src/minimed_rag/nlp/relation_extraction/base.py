"""Pseudocode relation extractor interface."""

from __future__ import annotations


class RelationExtractor:
    def extract(self, chunk, linked_mentions: list) -> list:
        raise NotImplementedError
