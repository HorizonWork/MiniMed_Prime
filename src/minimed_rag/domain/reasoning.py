"""Domain objects: reasoning tasks and paths."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class CandidateAnswer:
    text: str
    entity_id: str | None = None
    confidence: float = 0.0


@dataclass(slots=True)
class ReasoningTask:
    task_id: str
    question: str
    task_type: str
    options: list[CandidateAnswer] = field(default_factory=list)
    correct_answer: str | None = None
    answer_text: str | None = None
    linked_question_entities: list = field(default_factory=list)


@dataclass(slots=True)
class ReasoningPathStep:
    path_id: str
    step_index: int
    subject_entity_id: str
    predicate: str
    object_entity_id: str
    assertion_id: str
    edge_confidence: float
    direction: str = "forward"
    evidence_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReasoningPath:
    path_id: str
    task_id: str
    path_type: str
    start_entity_id: str
    end_entity_id: str
    task_type: str
    metapath_template_id: str = ""
    path_confidence: float = 0.0
    validity_label: str = "uncertain"
    graph_version: str = "kg_local"
    steps: list[ReasoningPathStep] = field(default_factory=list)
    negative_type: str | None = None
    max_conflict_score: float = 0.0
