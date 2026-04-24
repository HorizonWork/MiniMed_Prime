"""Pseudocode RAG context builder."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RetrievalContext:
    query: str
    plan: object
    graph_paths: list = field(default_factory=list)
    evidence_spans: list = field(default_factory=list)
    overall_confidence: float = 0.0


class ContextBuilder:
    def build(self, query: str, plan, results: list) -> RetrievalContext:
        graph_paths = [result for result in results if getattr(result, "kind", None) == "graph_path"]
        evidence_spans = [result for result in results if getattr(result, "kind", None) == "evidence"]
        confidence = max([getattr(result, "score", 0.0) for result in results], default=0.0)
        return RetrievalContext(query=query, plan=plan, graph_paths=graph_paths, evidence_spans=evidence_spans, overall_confidence=confidence)
