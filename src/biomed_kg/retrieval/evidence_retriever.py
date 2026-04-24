"""Pseudocode evidence retriever."""
from __future__ import annotations


class EvidenceRetriever:
    def __init__(self, evidence_repo):
        self.evidence_repo = evidence_repo

    def retrieve_for_graph_results(self, graph_results: list, top_k_per_assertion: int = 3) -> list:
        evidence = []
        for result in graph_results:
            for assertion_id in getattr(result, "assertion_ids", []):
                evidence.extend(self.evidence_repo.get_for_assertion(assertion_id)[:top_k_per_assertion])
        return evidence
