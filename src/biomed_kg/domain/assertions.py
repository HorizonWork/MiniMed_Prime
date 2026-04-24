"""Pseudocode domain object: canonical KG assertion."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Assertion:
    assertion_id: str
    subject_id: str
    predicate: str
    object_id: str
    polarity: str = "positive"
    directionality: str = "directed"
    confidence: float = 0.0
    source_reliability: float = 0.0
    entity_linking_confidence: float = 0.0
    relation_extraction_confidence: float = 0.0
    evidence_strength: float = 0.0
    consensus_score: float = 0.0
    conflict_score: float = 0.0
    recency_score: float = 0.0
    source_system: str = ""
    source_record_id: str = ""
    source_predicate: str | None = None
    source_release: str = ""
    graph_version: str = "kg_local"
    is_current: bool = True
