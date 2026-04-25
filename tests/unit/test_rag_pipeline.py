"""Unit tests for the Phase 4 RAG pipeline (KG augmentation path)."""

from __future__ import annotations

from dataclasses import dataclass, field

from minimed_rag.kg.graph_client import Neighbor, Node
from minimed_rag.nlp.linking.entity_linker import LinkedMention
from minimed_rag.retrieval.rag_pipeline import GraphFact, RAGPipeline


@dataclass
class FakeLinker:
    mentions: list[LinkedMention] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)

    def link_question(self, text: str) -> list[LinkedMention]:
        self.calls.append(text)
        return list(self.mentions)


@dataclass
class FakeGraphClient:
    neighbors_map: dict[str, list[Neighbor]] = field(default_factory=dict)
    calls: list[tuple[str, int, float]] = field(default_factory=list)

    def get_neighbors(
        self,
        key: str,
        depth: int = 1,
        limit: int = 50,
        min_confidence: float = 0.3,
    ) -> list[Neighbor]:
        self.calls.append((key, limit, min_confidence))
        return list(self.neighbors_map.get(key, []))


def _mention(text: str, cui: str, entity_id: str = "", conf: float = 0.9) -> LinkedMention:
    return LinkedMention(
        mention_text=text,
        expanded_text=text,
        entity_id=entity_id,
        concept_id=f"UMLS:{cui}" if cui else None,
        cui=cui,
        semantic_type="CHEMICAL",
        confidence=conf,
    )


def _neighbor(
    entity_id: str,
    name: str,
    predicate: str,
    confidence: float,
    polarity: str = "positive",
    source: str = "primekg",
    assertion_id: str = "ASSERT:x",
    direction: str = "out",
) -> Neighbor:
    return Neighbor(
        node=Node(labels=("Entity",), entity_id=entity_id, preferred_name=name),
        predicate=predicate,
        confidence=confidence,
        polarity=polarity,
        source_system=source,
        assertion_id=assertion_id,
        direction=direction,
    )


def test_pipeline_without_use_kg_skips_linking_and_graph():
    linker = FakeLinker(mentions=[_mention("m", "C1")])
    graph = FakeGraphClient()
    pipeline = RAGPipeline(graph_client=graph, question_entity_linker=linker)

    answer = pipeline.answer("What about metformin?", use_kg=False)

    assert answer.graph_paths == []
    assert answer.linked_mentions == []
    assert linker.calls == []
    assert graph.calls == []


def test_pipeline_use_kg_injects_sorted_graph_facts():
    mention = _mention("metformin", "C0025598", entity_id="KG:Drug:m")
    graph = FakeGraphClient(
        neighbors_map={
            "KG:Drug:m": [
                _neighbor("KG:Disease:a", "Diabetes", "treats", 0.9, assertion_id="ASSERT:1"),
                _neighbor(
                    "KG:GeneProtein:b",
                    "AMPK",
                    "targets",
                    0.85,
                    assertion_id="ASSERT:2",
                ),
                _neighbor(
                    "KG:Disease:c",
                    "Cancer",
                    "associated_with",
                    0.4,
                    assertion_id="ASSERT:3",
                ),
            ]
        }
    )
    linker = FakeLinker(mentions=[mention])
    pipeline = RAGPipeline(graph_client=graph, question_entity_linker=linker)

    answer = pipeline.answer("What does metformin treat?", use_kg=True, top_k=2, min_confidence=0.3)

    assert len(answer.graph_paths) == 2  # top_k=2
    assert [f.predicate for f in answer.graph_paths] == ["treats", "targets"]
    assert answer.graph_paths[0].confidence == 0.9
    assert answer.linked_mentions == [mention]
    # Average confidence is computed over injected facts
    assert abs(answer.confidence - ((0.9 + 0.85) / 2)) < 1e-9


def test_pipeline_dedupes_by_assertion_id():
    m1 = _mention("metformin", "C1", entity_id="KG:Drug:m")
    m2 = _mention("Metformin HCl", "C1", entity_id="KG:Drug:m")
    # Both mentions resolve to the same entity; neighbors should dedupe.
    shared = [_neighbor("KG:Disease:a", "Diabetes", "treats", 0.9)]
    graph = FakeGraphClient(neighbors_map={"KG:Drug:m": shared})
    linker = FakeLinker(mentions=[m1, m2])
    pipeline = RAGPipeline(graph_client=graph, question_entity_linker=linker)

    answer = pipeline.answer("?", use_kg=True)
    assert len(answer.graph_paths) == 1


def test_pipeline_handles_incoming_direction():
    """When the anchor is the OBJECT of the assertion, direction='in' and
    subject/object names should be swapped in the emitted GraphFact."""
    mention = _mention("diabetes", "C2", entity_id="KG:Disease:d")
    graph = FakeGraphClient(
        neighbors_map={
            "KG:Disease:d": [_neighbor("KG:Drug:m", "Metformin", "treats", 0.9, direction="in")]
        }
    )
    linker = FakeLinker(mentions=[mention])
    pipeline = RAGPipeline(graph_client=graph, question_entity_linker=linker)

    answer = pipeline.answer("What treats diabetes?", use_kg=True)
    fact: GraphFact = answer.graph_paths[0]
    assert fact.subject_name == "Metformin"  # drug is subject of "treats"
    assert fact.object_name == "diabetes"


def test_pipeline_use_kg_but_no_linked_entities_returns_no_facts():
    linker = FakeLinker(mentions=[])
    graph = FakeGraphClient(neighbors_map={"X": [_neighbor("KG:Disease:a", "D", "treats", 0.9)]})
    pipeline = RAGPipeline(graph_client=graph, question_entity_linker=linker)
    answer = pipeline.answer("?", use_kg=True)
    assert answer.graph_paths == []
    assert graph.calls == []


def test_build_prompt_contains_graph_facts_and_mention_info():
    pipeline = RAGPipeline()
    mentions = [_mention("metformin", "C1")]
    facts = [
        GraphFact(
            subject_key="KG:Drug:m",
            subject_name="Metformin",
            predicate="treats",
            object_key="KG:Disease:d",
            object_name="Diabetes",
            confidence=0.9,
            polarity="positive",
            source_system="primekg",
            assertion_id="ASSERT:1",
        ),
        GraphFact(
            subject_key="KG:Drug:a",
            subject_name="Aspirin",
            predicate="treats",
            object_key="KG:Disease:u",
            object_name="Ulcer",
            confidence=0.8,
            polarity="negative",
            source_system="primekg",
            assertion_id="ASSERT:2",
        ),
    ]
    prompt = pipeline.build_prompt("Q?", mentions, facts, [])

    assert "Q?" in prompt
    assert "metformin" in prompt
    assert "CUI=C1" in prompt
    assert "Metformin --[treats, conf=0.90, src=primekg]--> Diabetes" in prompt
    assert "(negated) Aspirin --[treats" in prompt


def test_pipeline_llm_integration_uses_generated_text():
    class FakeLLM:
        def generate(self, prompt: str) -> str:
            return "metformin treats type 2 diabetes"

    pipeline = RAGPipeline(llm=FakeLLM())
    answer = pipeline.answer("Q?", use_kg=False)
    assert answer.answer == "metformin treats type 2 diabetes"
