"""Pseudocode domain object: serialized training example."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SerializedTrainingExample:
    task_id: str
    question_tokens: list[int]
    option_tokens: list[int]
    entity_path: list[int]
    relation_path: list[int]
    edge_confidences: list[float]
    path_length: int
    task_type_id: int
    metapath_id: int
    label: int
    graph_version: str
    metadata: dict = field(default_factory=dict)
