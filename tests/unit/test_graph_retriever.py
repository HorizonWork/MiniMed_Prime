"""Unit tests for the Phase 5 GraphRetriever."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from minimed_rag.kg.graph_client import Edge, Neighbor, Node, TwoHopPath
from minimed_rag.retrieval.graph_retriever import GraphPath, GraphRetriever


def _node(entity_id: str, name: str) -> Node:
    return Node(labels=("Entity",), entity_id=entity_id, preferred_name=name)


def _neighbor_out(name: str, predicate: str, conf: float, *, eid="KG:N:" + "x") -> Neighbor:
    return Neighbor(
        node=_node(eid, name),
        predicate=predicate,
        confidence=conf,
        polarity="positive",
        source_system="primekg",
        assertion_id=f"ASSERT:{name}",
        direction="out",
    )


def _neighbor_in(name: str, predicate: str, conf: float, *, eid="KG:N:" + "x") -> Neighbor:
    return Neighbor(
        node=_node(eid, name),
        predicate=predicate,
        confidence=conf,
        polarity="positive",
        source_system="primekg",
        assertion_id=f"ASSERT:{name}",
        direction="in",
    )


@dataclass
class FakeGraphClient:
    one_hop_responses: dict
    two_hop_responses: dict | None = None

    def get_neighbors(self, key, depth=1, limit=10, min_confidence=0.5):
        return self.one_hop_responses.get(key, [])

    def get_two_hop_paths(self, key, *, limit=10, min_confidence=0.5):
        if self.two_hop_responses is None:
            return []
        return self.two_hop_responses.get(key, [])


def _mention(text: str, entity_id: str | None = None, cui: str | None = None):
    return SimpleNamespace(mention_text=text, entity_id=entity_id, cui=cui)


def test_one_hop_retrieval_builds_graph_paths():
    client = FakeGraphClient(
        one_hop_responses={
            "KG:Drug:m": [
                _neighbor_out("Type 2 Diabetes", "treats", 0.92, eid="KG:Disease:d"),
            ]
        }
    )
    retriever = GraphRetriever(client, hops=1, min_confidence=0.3)
    paths = retriever.retrieve([_mention("metformin", entity_id="KG:Drug:m")])

    assert len(paths) == 1
    p = paths[0]
    assert p.length == 1
    assert p.confidence == 0.92
    assert p.hops[0].predicate == "treats"
    assert p.hops[0].subject_name == "metformin"
    assert p.hops[0].object_name == "Type 2 Diabetes"
    assert p.serialize() == "metformin --TREATS--> Type 2 Diabetes"


def test_one_hop_anchor_as_object_keeps_assertion_direction():
    client = FakeGraphClient(
        one_hop_responses={
            "KG:Disease:d": [
                _neighbor_in("Metformin", "treats", 0.92, eid="KG:Drug:m"),
            ]
        }
    )
    retriever = GraphRetriever(client, hops=1, min_confidence=0.3)
    paths = retriever.retrieve([_mention("Type 2 Diabetes", entity_id="KG:Disease:d")])

    assert len(paths) == 1
    hop = paths[0].hops[0]
    assert hop.subject_name == "Metformin"
    assert hop.object_name == "Type 2 Diabetes"
    assert paths[0].serialize() == "Metformin --TREATS--> Type 2 Diabetes"


def test_two_hop_retrieval_uses_get_two_hop_paths():
    edge1 = Edge(assertion_id="A1", predicate="targets", subject_key="KG:Drug:m",
                 object_key="KG:Gene:g", confidence=0.85, polarity="positive", source_system="primekg")
    edge2 = Edge(assertion_id="A2", predicate="gene_associated_with_disease",
                 subject_key="KG:Gene:g", object_key="KG:Disease:d",
                 confidence=0.70, polarity="positive", source_system="primekg")
    two_hop = TwoHopPath(
        anchor=_node("KG:Drug:m", "metformin"),
        mid=_node("KG:Gene:g", "AMPK"),
        terminal=_node("KG:Disease:d", "T2D"),
        edge1=edge1,
        edge2=edge2,
    )
    client = FakeGraphClient(
        one_hop_responses={},
        two_hop_responses={"KG:Drug:m": [two_hop]},
    )
    retriever = GraphRetriever(client, hops=2, min_confidence=0.5)
    paths = retriever.retrieve([_mention("metformin", entity_id="KG:Drug:m")])

    assert len(paths) == 1
    assert paths[0].length == 2
    # weakest-link confidence
    assert paths[0].confidence == 0.70
    line = paths[0].serialize()
    assert "AMPK" in line
    assert "TARGETS" in line
    assert "GENE_ASSOCIATED_WITH_DISEASE" in line


def test_two_hop_mixed_direction_serializes_each_assertion_truthfully():
    edge1 = Edge(assertion_id="A1", predicate="targets", subject_key="KG:Drug:m",
                 object_key="KG:Gene:g", confidence=0.85, polarity="positive", source_system="primekg")
    edge2 = Edge(assertion_id="A2", predicate="associated_with", subject_key="KG:Disease:d",
                 object_key="KG:Gene:g", confidence=0.70, polarity="positive", source_system="primekg")
    two_hop = TwoHopPath(
        anchor=_node("KG:Drug:m", "metformin"),
        mid=_node("KG:Gene:g", "AMPK"),
        terminal=_node("KG:Disease:d", "T2D"),
        edge1=edge1,
        edge2=edge2,
    )
    client = FakeGraphClient(
        one_hop_responses={},
        two_hop_responses={"KG:Drug:m": [two_hop]},
    )
    retriever = GraphRetriever(client, hops=2, min_confidence=0.5)
    line = retriever.retrieve([_mention("metformin", entity_id="KG:Drug:m")])[0].serialize()

    assert line == "metformin --TARGETS--> AMPK ; T2D --ASSOCIATED_WITH--> AMPK"


def test_dedupe_skips_duplicate_one_hop_facts():
    client = FakeGraphClient(
        one_hop_responses={
            "KG:Drug:m": [
                _neighbor_out("Diabetes", "treats", 0.9, eid="KG:Disease:d"),
                _neighbor_out("Diabetes", "treats", 0.9, eid="KG:Disease:d"),  # duplicate
            ]
        }
    )
    retriever = GraphRetriever(client, hops=1)
    assert len(retriever.retrieve([_mention("metformin", entity_id="KG:Drug:m")])) == 1


def test_max_paths_caps_results():
    neighbors = [
        _neighbor_out(f"D{i}", "treats", 0.9 - i * 0.01, eid=f"KG:Disease:{i}")
        for i in range(50)
    ]
    client = FakeGraphClient(one_hop_responses={"KG:Drug:m": neighbors})
    retriever = GraphRetriever(client, hops=1, per_entity_limit=50, max_paths=5)
    paths = retriever.retrieve([_mention("metformin", entity_id="KG:Drug:m")])
    assert len(paths) == 5
    # sorted by confidence desc → first should be highest
    assert paths[0].confidence == 0.9


def test_skip_mentions_without_anchor_key():
    client = FakeGraphClient(one_hop_responses={})
    retriever = GraphRetriever(client, hops=1)
    # Mention with no entity_id and no cui
    assert retriever.retrieve([_mention("foo")]) == []


def test_serialize_paths_filters_empty():
    retriever = GraphRetriever(FakeGraphClient(one_hop_responses={}), hops=1)
    empty = GraphPath(hops=[], anchor_mention="x")
    assert retriever.serialize_paths([empty]) == []


def test_invalid_hops_value_raises():
    import pytest
    with pytest.raises(ValueError):
        GraphRetriever(FakeGraphClient(one_hop_responses={}), hops=3)
