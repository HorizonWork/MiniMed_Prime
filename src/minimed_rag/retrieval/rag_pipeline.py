"""RAG answer pipeline.

Phase 4 MVP scope:

- Question → linked mentions (via ``QuestionEntityLinker``).
- Linked CUIs → 1-hop graph neighbors (via ``GraphClient``).
- Neighbors serialised into prompt context, sorted by per-edge confidence.
- Hybrid text retrieval is optional; leave ``hybrid_retriever=None`` to run
  a KG-only pipeline for Phase 4 benchmarking.

Phase 5 will extend this with a ``QueryPlanner`` + metapath-aware
``GraphRetriever`` and a real LLM generator. Today the LLM is optional;
when absent, the pipeline returns a structured "no LLM" answer whose
``graph_paths`` field is still fully populated for inspection and
downstream evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GraphFact:
    subject_key: str
    subject_name: str
    predicate: str
    object_key: str
    object_name: str
    confidence: float
    polarity: str
    source_system: str
    assertion_id: str


@dataclass(slots=True)
class RAGAnswer:
    answer: str
    citations: list[str] = field(default_factory=list)
    graph_paths: list[GraphFact] = field(default_factory=list)
    linked_mentions: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    confidence: float = 0.0


class RAGPipeline:
    """Composable RAG pipeline with an optional KG-augmentation path."""

    def __init__(
        self,
        graph_client: Any = None,
        question_entity_linker: Any = None,
        hybrid_retriever: Any = None,
        llm: Any = None,
    ) -> None:
        self.graph_client = graph_client
        self.question_entity_linker = question_entity_linker
        self.hybrid_retriever = hybrid_retriever
        self.llm = llm

    def answer(
        self,
        user_query: str,
        use_kg: bool = False,
        top_k: int = 20,
        min_confidence: float = 0.3,
        per_entity_limit: int = 10,
    ) -> RAGAnswer:
        linked_mentions: list = []
        graph_paths: list[GraphFact] = []

        if use_kg and self.question_entity_linker and self.graph_client:
            linked_mentions = self.question_entity_linker.link_question(user_query)
            graph_paths = self._collect_graph_paths(
                linked_mentions,
                limit=per_entity_limit,
                min_confidence=min_confidence,
            )
            graph_paths.sort(key=lambda fact: fact.confidence, reverse=True)
            graph_paths = graph_paths[:top_k]

        evidence: list = []
        if self.hybrid_retriever is not None:
            hybrid_context = self.hybrid_retriever.retrieve(user_query)
            evidence = list(getattr(hybrid_context, "evidence_spans", []) or [])

        prompt = self.build_prompt(user_query, linked_mentions, graph_paths, evidence)

        if self.llm is not None:
            raw = self.llm.generate(prompt)
            answer_text = getattr(raw, "text", None) or (raw if isinstance(raw, str) else str(raw))
            citations = getattr(raw, "citations", []) or []
        else:
            answer_text = "[no LLM configured — graph_paths populated for inspection]"
            citations = []

        return RAGAnswer(
            answer=answer_text,
            citations=citations,
            graph_paths=graph_paths,
            linked_mentions=linked_mentions,
            evidence=evidence,
            confidence=self._aggregate_confidence(graph_paths),
        )

    def _collect_graph_paths(
        self,
        linked_mentions: list,
        limit: int,
        min_confidence: float,
    ) -> list[GraphFact]:
        facts: list[GraphFact] = []
        seen_assertions: set[str] = set()
        for mention in linked_mentions:
            anchor_key = mention.entity_id or mention.cui
            if not anchor_key:
                continue
            neighbors = self.graph_client.get_neighbors(
                anchor_key,
                depth=1,
                limit=limit,
                min_confidence=min_confidence,
            )
            for neighbor in neighbors:
                if neighbor.assertion_id in seen_assertions:
                    continue
                seen_assertions.add(neighbor.assertion_id)
                if neighbor.direction == "out":
                    subj_key, obj_key = anchor_key, neighbor.node.key
                    subj_name = mention.mention_text
                    obj_name = neighbor.node.preferred_name or neighbor.node.key
                else:
                    subj_key, obj_key = neighbor.node.key, anchor_key
                    subj_name = neighbor.node.preferred_name or neighbor.node.key
                    obj_name = mention.mention_text
                facts.append(
                    GraphFact(
                        subject_key=subj_key,
                        subject_name=subj_name,
                        predicate=neighbor.predicate,
                        object_key=obj_key,
                        object_name=obj_name,
                        confidence=neighbor.confidence,
                        polarity=neighbor.polarity,
                        source_system=neighbor.source_system,
                        assertion_id=neighbor.assertion_id,
                    )
                )
        return facts

    @staticmethod
    def _aggregate_confidence(graph_paths: list[GraphFact]) -> float:
        if not graph_paths:
            return 0.0
        total = sum(fact.confidence for fact in graph_paths)
        return total / len(graph_paths)

    def build_prompt(
        self,
        query: str,
        linked_mentions: list,
        graph_paths: list[GraphFact],
        evidence: list,
    ) -> str:
        sections = [
            "You are a biomedical assistant.",
            "Answer only using the provided evidence.",
            "If evidence is insufficient, say so.",
            "",
            f"Question:\n{query}",
        ]
        if linked_mentions:
            sections.append("\nLinked entities:")
            for mention in linked_mentions:
                sections.append(
                    f"  - {mention.mention_text} → "
                    f"CUI={mention.cui or 'N/A'} "
                    f"(type={mention.semantic_type or '-'}, "
                    f"conf={mention.confidence:.2f})"
                )
        if graph_paths:
            sections.append("\nKnowledge-graph facts (top-k by confidence):")
            for fact in graph_paths:
                marker = "(negated) " if fact.polarity == "negative" else ""
                sections.append(
                    f"  - {marker}{fact.subject_name} --["
                    f"{fact.predicate}, conf={fact.confidence:.2f}, "
                    f"src={fact.source_system}]--> {fact.object_name}"
                )
        if evidence:
            sections.append("\nTextual evidence:")
            for span in evidence[:10]:
                sections.append(f"  - {span}")
        sections.append(
            "\nInstructions:"
            "\n- Do not invent facts."
            "\n- Cite assertion IDs when referencing graph facts."
            "\n- Distinguish association from causation."
            "\n- Mention conflicts or negations if present."
        )
        return "\n".join(sections)
