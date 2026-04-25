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

    # ------------------------------------------------------------------
    # Phase 5: KG + text merge
    # ------------------------------------------------------------------

    @staticmethod
    def build_hybrid_context_text(
        graph_path_lines: list[str],
        text_results: list[RetrievalResult] | None,
        *,
        max_chars: int = 2500,
        graph_budget_ratio: float = 0.4,
    ) -> str:
        """Merge serialised KG paths + retrieved text chunks into a single context block.

        Format::

            [Graph Evidence]
            <serialised path 1>
            <serialised path 2>

            [Text Evidence]
            <chunk 1 text>

            <chunk 2 text>

        Budgeting: up to ``graph_budget_ratio * max_chars`` is reserved for
        the graph block (any unused share is given back to the text block).
        Empty sides are omitted (no dangling header).
        """
        if not (0.0 <= graph_budget_ratio <= 1.0):
            raise ValueError("graph_budget_ratio must be in [0, 1]")

        clean_graph_lines = [line.strip() for line in (graph_path_lines or []) if line and line.strip()]
        text_results = text_results or []

        graph_budget = int(max_chars * graph_budget_ratio) if clean_graph_lines else 0
        text_budget = max_chars - graph_budget

        graph_block = ""
        if clean_graph_lines:
            graph_lines: list[str] = []
            graph_total = 0
            header = "[Graph Evidence]\n"
            for line in clean_graph_lines:
                if graph_total + len(line) + 1 > graph_budget:
                    break
                graph_lines.append(line)
                graph_total += len(line) + 1
            if graph_lines:
                graph_block = header + "\n".join(graph_lines)
                # If we underran the graph budget, give the slack back to text.
                text_budget = max_chars - len(graph_block)

        text_block = ""
        if text_results and text_budget > 0:
            inner = ContextBuilder.build_context_text(text_results, max_chars=text_budget)
            if inner.strip():
                text_block = "[Text Evidence]\n" + inner

        if graph_block and text_block:
            return f"{graph_block}\n\n{text_block}"
        return graph_block or text_block
