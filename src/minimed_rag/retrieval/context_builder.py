from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minimed_rag.retrieval.bm25_retriever import RetrievalResult


@dataclass(slots=True)
class RetrievalContext:
    query: str
    plan: object
    graph_paths: list = field(default_factory=list)
    evidence_spans: list = field(default_factory=list)
    overall_confidence: float = 0.0


class ContextBuilder:
    """Build retrieval context for the hybrid RAG pipeline."""

    def build(self, query: str, plan, results: list) -> RetrievalContext:
        graph_paths = [r for r in results if getattr(r, "kind", None) == "graph_path"]
        evidence_spans = [r for r in results if getattr(r, "kind", None) == "evidence"]
        confidence = max((getattr(r, "score", 0.0) for r in results), default=0.0)
        return RetrievalContext(
            query=query,
            plan=plan,
            graph_paths=graph_paths,
            evidence_spans=evidence_spans,
            overall_confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Lightweight helper for benchmark / MCQ evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def build_context_text(
        results: list[RetrievalResult],
        max_chars: int = 2000,
    ) -> str:
        """Concatenate retrieved chunk texts into a single context string.

        Truncates at *max_chars* so it fits inside the prompt budget.
        """
        parts: list[str] = []
        total = 0
        for result in results:
            snippet = result.chunk.text.strip()
            if not snippet:
                continue
            if total + len(snippet) > max_chars:
                remaining = max_chars - total
                if remaining > 50:
                    parts.append(snippet[:remaining])
                break
            parts.append(snippet)
            total += len(snippet)
        return "\n\n".join(parts)
