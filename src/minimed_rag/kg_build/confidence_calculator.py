"""Pseudocode assertion confidence calculator."""

from __future__ import annotations

from minimed_rag.domain.assertions import Assertion


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class ConfidenceCalculator:
    def __init__(self, policy):
        self.policy = policy

    def calculate(self, assertion: Assertion) -> Assertion:
        source_reliability = self.get_source_reliability(assertion)
        entity_linking_confidence = self.get_entity_linking_confidence(assertion)
        relation_extraction_confidence = self.get_relation_extraction_confidence(assertion)
        evidence_strength = self.get_evidence_strength(assertion)
        consensus_score = self.get_consensus_score(assertion)
        conflict_score = self.get_conflict_score(assertion)
        recency_score = self.get_recency_score(assertion)
        w = self.policy.weights
        confidence = (
            w.get("source_reliability", 0.0) * source_reliability
            + w.get("entity_linking_confidence", 0.0) * entity_linking_confidence
            + w.get("relation_extraction_confidence", 0.0) * relation_extraction_confidence
            + w.get("evidence_strength", 0.0) * evidence_strength
            + w.get("consensus_score", 0.0) * consensus_score
            + w.get("recency_score", 0.0) * recency_score
            - abs(w.get("conflict_score", 0.0)) * conflict_score
        )
        assertion.source_reliability = source_reliability
        assertion.entity_linking_confidence = entity_linking_confidence
        assertion.relation_extraction_confidence = relation_extraction_confidence
        assertion.evidence_strength = evidence_strength
        assertion.consensus_score = consensus_score
        assertion.conflict_score = conflict_score
        assertion.recency_score = recency_score
        assertion.confidence = clamp(confidence, 0.0, 1.0)
        return assertion

    def get_source_reliability(self, assertion: Assertion) -> float:
        return float(self.policy.source_reliability.get(assertion.source_system, 0.5))

    def get_entity_linking_confidence(self, assertion: Assertion) -> float:
        return assertion.entity_linking_confidence or 1.0

    def get_relation_extraction_confidence(self, assertion: Assertion) -> float:
        return assertion.relation_extraction_confidence or 1.0

    def get_evidence_strength(self, assertion: Assertion) -> float:
        return assertion.evidence_strength or 0.5

    def get_consensus_score(self, assertion: Assertion) -> float:
        return assertion.consensus_score or 0.5

    def get_conflict_score(self, assertion: Assertion) -> float:
        return assertion.conflict_score

    def get_recency_score(self, assertion: Assertion) -> float:
        return assertion.recency_score or 0.5
