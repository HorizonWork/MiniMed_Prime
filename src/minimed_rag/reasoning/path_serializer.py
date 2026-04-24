"""Pseudocode path serializer for TRM."""

from __future__ import annotations


class PathSerializer:
    def __init__(
        self,
        entity_vocab,
        relation_vocab,
        task_type_vocab,
        metapath_vocab,
        question_encoder,
        option_encoder,
    ):
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.task_type_vocab = task_type_vocab
        self.metapath_vocab = metapath_vocab
        self.question_encoder = question_encoder
        self.option_encoder = option_encoder

    def serialize_examples(self, task, positive_paths, negative_paths, graph_version: str):
        examples = []
        for path in positive_paths:
            examples.append(self.serialize_path(task, path, 1, graph_version))
        for path in negative_paths:
            examples.append(self.serialize_path(task, path, 0, graph_version))
        return examples

    def serialize_path(self, task, path, label: int, graph_version: str):
        entity_tokens = []
        relation_tokens = []
        for step in path.steps:
            entity_tokens.append(self.entity_vocab.get_id(step.subject_entity_id))
            relation_tokens.append(self.relation_vocab.get_id(step.predicate))
        entity_tokens.append(self.entity_vocab.get_id(path.steps[-1].object_entity_id))
        return {
            "task_id": task.task_id,
            "question_tokens": self.question_encoder.encode(task.question),
            "option_tokens": self.option_encoder.encode(task.answer_text),
            "entity_path": entity_tokens,
            "relation_path": relation_tokens,
            "edge_confidences": [step.edge_confidence for step in path.steps],
            "path_length": len(path.steps),
            "task_type_id": self.task_type_vocab.get_id(task.task_type),
            "metapath_id": self.metapath_vocab.get_id(path.metapath_template_id),
            "label": label,
            "graph_version": graph_version,
        }
