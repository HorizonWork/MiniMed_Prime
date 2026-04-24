"""Pseudocode reasoning dataset builder."""
from __future__ import annotations


class ReasoningDatasetBuilder:
    def __init__(self, entity_linker, ner, path_finder, path_scorer, path_pruner, negative_sampler, path_serializer, training_repo):
        self.entity_linker = entity_linker
        self.ner = ner
        self.path_finder = path_finder
        self.path_scorer = path_scorer
        self.path_pruner = path_pruner
        self.negative_sampler = negative_sampler
        self.path_serializer = path_serializer
        self.training_repo = training_repo

    def build(self, graph_version: str, dataset_name: str) -> None:
        tasks = self.load_reasoning_tasks(dataset_name)
        for task in tasks:
            linked_question_entities = self.entity_linker.link_mentions(task.question, self.ner.extract(task.question))
            linked_answer_entities = self.link_answers(task)
            if not linked_question_entities or not linked_answer_entities:
                self.mark_unusable(task, "entity_linking_failed")
                continue
            candidate_paths = self.path_finder.find_candidate_paths(linked_question_entities, linked_answer_entities, task.task_type, graph_version)
            scored_paths = self.path_scorer.score(candidate_paths, task)
            positive_paths = self.path_pruner.select_positive_paths(scored_paths)
            negative_paths = self.negative_sampler.generate(task, positive_paths, graph_version)
            examples = self.path_serializer.serialize_examples(task, positive_paths, negative_paths, graph_version)
            self.training_repo.write_examples(examples)

    def load_reasoning_tasks(self, dataset_name: str):
        return []

    def link_answers(self, task):
        return []

    def mark_unusable(self, task, reason: str) -> None:
        return None
