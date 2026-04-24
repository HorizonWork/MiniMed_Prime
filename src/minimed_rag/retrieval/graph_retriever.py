"""Pseudocode graph retriever."""

from __future__ import annotations


class GraphRetriever:
    def __init__(self, neo4j, path_scorer, path_pruner):
        self.neo4j = neo4j
        self.path_scorer = path_scorer
        self.path_pruner = path_pruner

    def retrieve(self, plan) -> list:
        results = []
        for template in plan.metapath_templates:
            if template.name == "treatment_direct":
                results.extend(self.retrieve_treatment_direct(plan, template))
            elif template.name == "treatment_mechanism":
                results.extend(self.retrieve_treatment_mechanism(plan, template))
            elif template.name == "adverse_effect_direct":
                results.extend(self.retrieve_adverse_effect_direct(plan, template))
            elif template.name == "mechanism_drug_pathway_disease":
                results.extend(self.retrieve_mechanism_paths(plan, template))
        scored = self.path_scorer.score(results, plan)
        pruned = self.path_pruner.prune(scored, plan)
        return sorted(pruned, key=lambda result: result.score, reverse=True)

    def retrieve_treatment_direct(self, plan, template):
        disease = plan.get_entity_by_type("Disease")
        return self.neo4j.query(
            """
            MATCH (a:Assertion {predicate: "treats", is_current: true})-[:OBJECT]->(d:Entity {entity_id: $disease_id})
            MATCH (a)-[:SUBJECT]->(drug:Entity)
            WHERE a.confidence >= $min_confidence
            RETURN drug, a, d
            ORDER BY a.confidence DESC
            LIMIT $limit
            """,
            {
                "disease_id": disease.entity_id,
                "min_confidence": template.min_edge_confidence,
                "limit": 50,
            },
        )

    def retrieve_treatment_mechanism(self, plan, template):
        return []

    def retrieve_adverse_effect_direct(self, plan, template):
        adverse_event = plan.get_entity_by_type(["AdverseEvent", "Finding", "Symptom"])
        return self.neo4j.query(
            """
            MATCH (a:Assertion {predicate: "causes_adverse_event", is_current: true})-[:OBJECT]->(ae:Entity {entity_id: $ae_id})
            MATCH (a)-[:SUBJECT]->(drug:Entity)
            WHERE a.confidence >= $min_confidence
            RETURN drug, a, ae
            ORDER BY a.confidence DESC
            LIMIT $limit
            """,
            {
                "ae_id": adverse_event.entity_id,
                "min_confidence": template.min_edge_confidence,
                "limit": 50,
            },
        )

    def retrieve_mechanism_paths(self, plan, template):
        return []
