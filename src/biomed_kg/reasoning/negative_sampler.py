"""Pseudocode negative sampler for TRM path training."""
from __future__ import annotations

import copy
import random


class NegativeSampler:
    def __init__(self, entity_linker, path_finder, predicate_registry, entity_repo):
        self.entity_linker = entity_linker
        self.path_finder = path_finder
        self.predicate_registry = predicate_registry
        self.entity_repo = entity_repo

    def generate(self, task, positive_paths, graph_version: str):
        negatives = []
        negatives += self.generate_wrong_answer_paths(task, graph_version)
        negatives += self.generate_reversed_direction_paths(positive_paths)
        negatives += self.generate_generic_hub_paths(task, graph_version)
        negatives += self.generate_relation_corruption_paths(positive_paths)
        negatives += self.generate_entity_corruption_paths(positive_paths)
        negatives += self.generate_unsupported_paths(task, graph_version)
        return self.balance_negatives(negatives, max_per_task=10)

    def generate_wrong_answer_paths(self, task, graph_version: str):
        paths = []
        wrong_answers = [option for option in task.options if option.text != task.correct_answer]
        for wrong_answer in wrong_answers:
            linked_wrong = self.entity_linker.link_answer(wrong_answer)
            if not linked_wrong:
                continue
            candidate_paths = self.path_finder.find_candidate_paths(task.linked_question_entities, linked_wrong, task.task_type, graph_version)
            for path in candidate_paths:
                path.path_type = "hard_negative"
                path.validity_label = "invalid"
                path.negative_type = "wrong_answer_path"
                paths.append(path)
        return paths

    def generate_reversed_direction_paths(self, positive_paths):
        negatives = []
        for path in positive_paths:
            reversed_steps = []
            for step in reversed(path.steps):
                if self.predicate_registry.is_directional(step.predicate):
                    reversed_step = copy.copy(step)
                    reversed_step.subject_entity_id = step.object_entity_id
                    reversed_step.object_entity_id = step.subject_entity_id
                    reversed_step.direction = "reversed"
                    reversed_steps.append(reversed_step)
            if reversed_steps:
                negative = copy.copy(path)
                negative.path_type = "hard_negative"
                negative.validity_label = "invalid"
                negative.negative_type = "reversed_causal_direction"
                negative.steps = reversed_steps
                negatives.append(negative)
        return negatives

    def generate_relation_corruption_paths(self, positive_paths):
        negatives = []
        for path in positive_paths:
            corrupted = copy.deepcopy(path)
            if not corrupted.steps:
                continue
            step = random.choice(corrupted.steps)
            step.predicate = self.predicate_registry.sample_incompatible_predicate(self.entity_repo.get_type(step.subject_entity_id), self.entity_repo.get_type(step.object_entity_id), step.predicate)
            corrupted.path_type = "hard_negative"
            corrupted.validity_label = "invalid"
            corrupted.negative_type = "relation_type_mismatch"
            negatives.append(corrupted)
        return negatives

    def generate_generic_hub_paths(self, task, graph_version: str):
        return []

    def generate_entity_corruption_paths(self, positive_paths):
        return []

    def generate_unsupported_paths(self, task, graph_version: str):
        return []

    def balance_negatives(self, negatives, max_per_task: int):
        return negatives[:max_per_task]
