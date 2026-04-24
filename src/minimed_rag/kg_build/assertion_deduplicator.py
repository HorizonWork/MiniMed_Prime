"""Pseudocode assertion deduplication."""

from __future__ import annotations


class AssertionDeduplicator:
    def deduplicate(self, assertions: list) -> list:
        grouped = {}
        for assertion in assertions:
            key = (
                assertion.subject_id,
                assertion.predicate,
                assertion.object_id,
                assertion.polarity,
            )
            grouped.setdefault(key, []).append(assertion)
        return [self.merge_group(group) for group in grouped.values()]

    def merge_group(self, group: list):
        winner = max(group, key=lambda item: item.confidence)
        winner.consensus_score = min(1.0, len(group) / 3)
        return winner
