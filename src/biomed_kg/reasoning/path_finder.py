"""Pseudocode task-specific path finder."""
from __future__ import annotations


class PathFinder:
    def __init__(self, neo4j, metapath_registry):
        self.neo4j = neo4j
        self.metapath_registry = metapath_registry

    def find_candidate_paths(self, question_entities: list, answer_entities: list, task_type: str, graph_version: str) -> list:
        templates = self.metapath_registry.get_for_task(task_type)
        all_paths = []
        for question_entity in question_entities:
            for answer_entity in answer_entities:
                for template in templates:
                    all_paths.extend(self.find_paths_by_template(question_entity, answer_entity, template, graph_version))
        return all_paths

    def find_paths_by_template(self, start_entity, end_entity, template, graph_version: str):
        if len(template.pattern) == 1:
            return self.find_one_hop_paths(start_entity, end_entity, template, graph_version)
        if len(template.pattern) == 2:
            return self.find_two_hop_paths(start_entity, end_entity, template, graph_version)
        if len(template.pattern) == 3:
            return self.find_three_hop_paths(start_entity, end_entity, template, graph_version)
        return self.find_constrained_variable_paths(start_entity, end_entity, template, graph_version)

    def find_one_hop_paths(self, start, end, template, graph_version: str):
        step = template.pattern[0]
        query = """
        MATCH (s:Entity {entity_id: $start_id})
        MATCH (e:Entity {entity_id: $end_id})
        MATCH (a:Assertion {predicate: $predicate, is_current: true})-[:SUBJECT]->(s)
        MATCH (a)-[:OBJECT]->(e)
        WHERE a.confidence >= $min_conf AND a.graph_version = $graph_version
        RETURN s, a, e
        LIMIT $limit
        """
        return [self.row_to_path(row, template) for row in self.neo4j.query(query, {"start_id": start.entity_id, "end_id": end.entity_id, "predicate": step.predicate, "min_conf": template.min_edge_confidence, "graph_version": graph_version, "limit": 100})]

    def find_two_hop_paths(self, start, end, template, graph_version: str):
        step1 = template.pattern[0]
        step2 = template.pattern[1]
        query = """
        MATCH (s:Entity {entity_id: $start_id})
        MATCH (e:Entity {entity_id: $end_id})
        MATCH (a1:Assertion {predicate: $p1, is_current: true})-[:SUBJECT]->(s)
        MATCH (a1)-[:OBJECT]->(mid:Entity)
        MATCH (a2:Assertion {predicate: $p2, is_current: true})-[:SUBJECT]->(mid)
        MATCH (a2)-[:OBJECT]->(e)
        WHERE a1.confidence >= $min_conf
          AND a2.confidence >= $min_conf
          AND mid.entity_type = $mid_type
          AND a1.graph_version = $graph_version
          AND a2.graph_version = $graph_version
        RETURN s, a1, mid, a2, e
        LIMIT $limit
        """
        rows = self.neo4j.query(query, {"start_id": start.entity_id, "end_id": end.entity_id, "p1": step1.predicate, "p2": step2.predicate, "mid_type": step1.object_type, "min_conf": template.min_edge_confidence, "graph_version": graph_version, "limit": 100})
        return [self.row_to_path(row, template) for row in rows]

    def find_three_hop_paths(self, start, end, template, graph_version: str):
        return []

    def find_constrained_variable_paths(self, start, end, template, graph_version: str):
        return []

    def row_to_path(self, row, template):
        return row
