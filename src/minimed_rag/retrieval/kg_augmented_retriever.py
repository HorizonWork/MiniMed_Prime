"""Wrapper that combines an inner text retriever with a graph retriever.

Returned by ``cli/evaluate.py::_build_kg_augmented_retriever`` and consumed
by ``benchmark/evaluator.py``. Exposes a single ``retrieve_with_kg(query)``
method that returns a tuple of (text RetrievalResults, serialised graph
path lines, raw GraphPath objects). Keeping the raw paths on the side
lets the evaluator hand them to the conflict reporter without re-running
retrieval.

When ``text_retriever`` is ``None`` the wrapper degrades to graph-only;
when ``graph_retriever`` is ``None`` it degrades to text-only — both are
legal ablation modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from minimed_rag.retrieval.query_planner import QueryPlan, decide_strategy

if TYPE_CHECKING:
    from minimed_rag.retrieval.bm25_retriever import RetrievalResult
    from minimed_rag.retrieval.graph_retriever import GraphPath, GraphRetriever


@dataclass(slots=True)
class KGAugmentedResult:
    text_results: list[RetrievalResult] = field(default_factory=list)
    graph_paths: list[GraphPath] = field(default_factory=list)
    graph_path_lines: list[str] = field(default_factory=list)
    linked_mentions: list = field(default_factory=list)
    query_plan: QueryPlan = QueryPlan.TEXT_ONLY
    text_latency_s: float = 0.0
    graph_latency_s: float = 0.0


class KGAugmentedRetriever:
    """Composes a text retriever + graph retriever + question entity linker."""

    name: str

    def __init__(
        self,
        *,
        text_retriever: Any | None,
        graph_retriever: GraphRetriever | None,
        question_linker: Any | None,
        name: str = "kg_augmented",
        text_top_k: int | None = None,
    ) -> None:
        self.text_retriever = text_retriever
        self.graph_retriever = graph_retriever
        self.question_linker = question_linker
        self.name = name
        self.text_top_k = text_top_k

    @property
    def has_text(self) -> bool:
        return self.text_retriever is not None

    @property
    def has_graph(self) -> bool:
        return self.graph_retriever is not None and self.question_linker is not None

    def retrieve_with_kg(self, query: str) -> KGAugmentedResult:
        import time

        linked_mentions: list = []
        plan = QueryPlan.TEXT_ONLY

        if self.has_graph:
            t0 = time.perf_counter()
            try:
                linked_mentions = self.question_linker.link_question(query)
            except Exception:
                linked_mentions = []
            graph_link_latency = time.perf_counter() - t0
        else:
            graph_link_latency = 0.0

        plan = decide_strategy(
            linked_mentions,
            text_available=self.has_text,
            graph_available=self.has_graph,
        )

        text_results: list[RetrievalResult] = []
        text_latency = 0.0
        if self.text_retriever is not None and plan in (QueryPlan.TEXT_ONLY, QueryPlan.HYBRID):
            t0 = time.perf_counter()
            try:
                if self.text_top_k is not None:
                    text_results = list(self.text_retriever.retrieve(query, self.text_top_k))
                else:
                    text_results = list(self.text_retriever.retrieve(query))
            except Exception:
                text_results = []
            text_latency = time.perf_counter() - t0

        graph_paths: list = []
        graph_lines: list[str] = []
        graph_latency = graph_link_latency
        if self.has_graph and plan in (QueryPlan.GRAPH_ONLY, QueryPlan.HYBRID):
            t0 = time.perf_counter()
            try:
                graph_paths = self.graph_retriever.retrieve(linked_mentions)
                graph_lines = self.graph_retriever.serialize_paths(graph_paths)
            except Exception:
                graph_paths = []
                graph_lines = []
            graph_latency = time.perf_counter() - t0

        return KGAugmentedResult(
            text_results=text_results,
            graph_paths=graph_paths,
            graph_path_lines=graph_lines,
            linked_mentions=linked_mentions,
            query_plan=plan,
            text_latency_s=text_latency,
            graph_latency_s=graph_latency,
        )

    # Compatibility shim so legacy callers can still invoke ``.retrieve(query)``
    # and only get the text side back (drops graph context silently).
    def retrieve(self, query: str, k: int | None = None):
        result = self.retrieve_with_kg(query)
        return result.text_results[: k or len(result.text_results)]


__all__ = ["KGAugmentedResult", "KGAugmentedRetriever"]
