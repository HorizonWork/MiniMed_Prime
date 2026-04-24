"""Pseudocode hybrid graph/vector/BM25/evidence retriever."""
from __future__ import annotations


class HybridRetriever:
    def __init__(self, query_planner, graph_retriever, vector_retriever, bm25_retriever, evidence_retriever, reranker, context_builder, graph_version_provider):
        self.query_planner = query_planner
        self.graph_retriever = graph_retriever
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.evidence_retriever = evidence_retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.graph_version_provider = graph_version_provider

    def retrieve(self, query: str):
        plan = self.query_planner.plan(query)
        graph_results = self.graph_retriever.retrieve(plan)
        graph_version = self.graph_version_provider()
        vector_results = self.vector_retriever.retrieve(query, {"entity_ids": [entity.entity_id for entity in plan.linked_entities], "graph_version": graph_version}, 100)
        bm25_results = self.bm25_retriever.retrieve(query, {"graph_version": graph_version}, 100)
        evidence_results = self.evidence_retriever.retrieve_for_graph_results(graph_results, 3)
        merged = self.merge_results(graph_results, vector_results, bm25_results, evidence_results)
        reranked = self.reranker.rerank(query=query, results=merged)
        return self.context_builder.build(query=query, plan=plan, results=reranked[:20])

    def merge_results(self, graph_results, vector_results, bm25_results, evidence_results):
        merged = {}
        for result in graph_results:
            merged[result.key] = result.with_score_component("graph", result.score)
        for result in vector_results:
            merged.setdefault(result.key, result)
            merged[result.key].add_score_component("vector", result.score)
        for result in bm25_results:
            merged.setdefault(result.key, result)
            merged[result.key].add_score_component("bm25", result.score)
        for result in evidence_results:
            merged.setdefault(result.key, result)
            merged[result.key].add_score_component("evidence", result.score)
        return list(merged.values())
