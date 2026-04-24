"""Pseudocode conflict detection."""

from __future__ import annotations

from collections import defaultdict


class ConflictDetector:
    def __init__(self, assertion_repo, conflict_repo=None):
        self.assertion_repo = assertion_repo
        self.conflict_repo = conflict_repo

    def run(self) -> None:
        grouped = defaultdict(list)
        for assertion in self.assertion_repo.iter_current_assertions():
            grouped[(assertion.subject_id, assertion.object_id)].append(assertion)
        for (subject_id, object_id), group in grouped.items():
            predicates = {assertion.predicate for assertion in group}
            conflicts = self.detect_predicate_conflicts(predicates)
            if conflicts:
                for assertion in group:
                    assertion.conflict_score = self.score_conflict(assertion, group)
                    self.assertion_repo.update(assertion)
                if self.conflict_repo:
                    self.conflict_repo.write(
                        subject_id=subject_id,
                        object_id=object_id,
                        conflict_predicates=list(predicates),
                        conflict_type="predicate_conflict",
                    )

    def detect_predicate_conflicts(self, predicates: set[str]) -> list[str]:
        rules = [
            ("treats", "contraindicated_for"),
            ("causes", "prevents"),
            ("increases_risk_of", "decreases_risk_of"),
        ]
        return [
            f"{left}_vs_{right}"
            for left, right in rules
            if left in predicates and right in predicates
        ]

    def score_conflict(self, assertion, group) -> float:
        return min(1.0, 0.25 * (len({item.predicate for item in group}) - 1))
