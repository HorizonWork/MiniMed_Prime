"""Conflict detection helpers for KG and KG-vs-text reporting."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from minimed_rag.kg_build.runtime_conflict_reporter import (
    Conflict,
    RuntimeConflictReporter,
    write_conflicts_jsonl,
)

CONFLICTING_PREDICATES = (
    ("treats", "contraindicated_for"),
    ("causes", "prevents"),
    ("increases_risk_of", "decreases_risk_of"),
)


@dataclass(slots=True)
class AssertionConflict:
    subject_id: str
    object_id: str
    rules: list[str]
    assertion_ids: list[str]
    conflict_type: str = "predicate_conflict"


class ConflictDetector:
    """Detect self-conflicting current graph assertions.

    Runtime graph-vs-text contradiction checks are provided by
    ``RuntimeConflictReporter`` from this module for the Phase 5 benchmark
    path. This class keeps the offline assertion-repo API usable.
    """

    def __init__(self, assertion_repo, conflict_repo=None):
        self.assertion_repo = assertion_repo
        self.conflict_repo = conflict_repo

    def run(self) -> list[AssertionConflict]:
        grouped = defaultdict(list)
        for assertion in self.assertion_repo.iter_current_assertions():
            grouped[(assertion.subject_id, assertion.object_id)].append(assertion)
        found: list[AssertionConflict] = []
        for (subject_id, object_id), group in grouped.items():
            rules = self.detect_assertion_conflicts(group)
            if not rules:
                continue
            for assertion in group:
                assertion.conflict_score = self.score_conflict(assertion, group)
                self.assertion_repo.update(assertion)
            record = AssertionConflict(
                subject_id=subject_id,
                object_id=object_id,
                rules=rules,
                assertion_ids=[assertion.assertion_id for assertion in group],
            )
            found.append(record)
            if self.conflict_repo:
                payload = {
                    "subject_id": subject_id,
                    "object_id": object_id,
                    "conflict_predicates": [assertion.predicate for assertion in group],
                    "conflict_type": record.conflict_type,
                }
                try:
                    self.conflict_repo.write(**payload, conflict_rules=rules)
                except TypeError:
                    self.conflict_repo.write(**payload)
        return found

    def detect_predicate_conflicts(self, predicates: set[str]) -> list[str]:
        return [
            f"{left}_vs_{right}"
            for left, right in CONFLICTING_PREDICATES
            if left in predicates and right in predicates
        ]

    def detect_assertion_conflicts(self, assertions: list) -> list[str]:
        predicates = {assertion.predicate for assertion in assertions}
        rules = self.detect_predicate_conflicts(predicates)
        polarity_by_predicate: dict[str, set[str]] = defaultdict(set)
        for assertion in assertions:
            polarity_by_predicate[assertion.predicate].add(
                getattr(assertion, "polarity", "positive") or "positive"
            )
        for predicate, polarities in polarity_by_predicate.items():
            if {"positive", "negative"}.issubset(polarities):
                rules.append(f"{predicate}_positive_vs_negative")
        return rules

    def score_conflict(self, assertion, group) -> float:
        _ = assertion
        rules = self.detect_assertion_conflicts(group)
        return min(1.0, 0.25 * len(rules))


__all__ = [
    "AssertionConflict",
    "Conflict",
    "ConflictDetector",
    "RuntimeConflictReporter",
    "write_conflicts_jsonl",
]
