"""Pseudocode RAG answer pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RAGAnswer:
    answer: str
    citations: list[str] = field(default_factory=list)
    graph_paths: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    confidence: float = 0.0


class RAGPipeline:
    def __init__(self, hybrid_retriever, llm):
        self.hybrid_retriever = hybrid_retriever
        self.llm = llm

    def answer(self, user_query: str) -> RAGAnswer:
        context = self.hybrid_retriever.retrieve(user_query)
        prompt = self.build_prompt(user_query, context)
        raw_answer = self.llm.generate(prompt)
        grounded_answer = self.verify_grounding(answer=raw_answer, evidence=context.evidence_spans)
        return RAGAnswer(grounded_answer.text, grounded_answer.citations, context.graph_paths, context.evidence_spans, context.overall_confidence)

    def build_prompt(self, query, context) -> str:
        return f"""
You are a biomedical assistant.
Answer only using the provided evidence.
If evidence is insufficient, say so.

Question:
{query}

Graph paths:
{context.graph_paths}

Evidence:
{context.evidence_spans}

Instructions:
- Do not invent facts.
- Cite evidence IDs.
- Distinguish association from causation.
- Mention conflicts if present.
""".strip()

    def verify_grounding(self, answer, evidence):
        return answer
