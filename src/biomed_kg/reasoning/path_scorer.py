"""Pseudocode reasoning path scorer."""
from __future__ import annotations


class PathScorer:
    def __init__(self, metapath_registry, graph_stats):
        self.metapath_registry = metapath_registry
        self.graph_stats = graph_stats

    def score(self, paths: list, task=None) -> list:
        for path in paths:
            edge_confidences = [step.edge_confidence for step in path.steps]
            min_edge_confidence = min(edge_confidences) if edge_confidences else 0.0
            metapath_prior = self.metapath_registry.get(path.metapath_template_id).priority
            evidence_coverage = self.calculate_evidence_coverage(path)
            hub_penalty = self.calculate_hub_penalty(path)
            directionality_score = self.calculate_directionality_score(path)
            associated_with_penalty = self.calculate_associated_with_penalty(path)
            entity_relevance = self.calculate_entity_relevance(path, task)
            path.path_confidence = min_edge_confidence * metapath_prior * evidence_coverage * directionality_score * entity_relevance * (1.0 - hub_penalty) * (1.0 - associated_with_penalty)
        return paths

    def calculate_evidence_coverage(self, path) -> float:
        if not path.steps:
            return 0.0
        return sum(1 for step in path.steps if step.evidence_ids) / len(path.steps)

    def calculate_hub_penalty(self, path) -> float:
        penalty = 0.0
        for step in path.steps:
            for entity_id in [step.subject_entity_id, step.object_entity_id]:
                degree = self.graph_stats.get_degree(entity_id)
                if degree > 10000:
                    penalty += 0.20
                elif degree > 1000:
                    penalty += 0.10
                elif degree > 100:
                    penalty += 0.03
        return min(penalty, 0.70)

    def calculate_directionality_score(self, path) -> float:
        return 0.5 if any(step.direction == "reversed" for step in path.steps) else 1.0

    def calculate_associated_with_penalty(self, path) -> float:
        count = sum(1 for step in path.steps if step.predicate == "associated_with")
        if count == 0:
            return 0.0
        if count == 1:
            return 0.25
        return 0.60

    def calculate_entity_relevance(self, path, task=None) -> float:
        return 1.0
