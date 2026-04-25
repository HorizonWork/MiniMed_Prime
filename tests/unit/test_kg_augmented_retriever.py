"""Unit tests for the KGAugmentedRetriever wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from minimed_rag.retrieval.graph_retriever import GraphFact, GraphPath
from minimed_rag.retrieval.kg_augmented_retriever import KGAugmentedRetriever


@dataclass
class _Chunk:
    text: str
    id: str = "c1"
    source: str = "x"


@dataclass
class _Result:
    chunk: _Chunk
    score: float


class _FakeText:
    def __init__(self, results):
        self._results = results

    def retrieve(self, query, k=None):
        return list(self._results)


class _FakeLinker:
    def __init__(self, mentions):
        self._mentions = mentions

    def link_question(self, query):
        return list(self._mentions)


class _FakeGraphRetriever:
    def __init__(self, paths):
        self._paths = paths
        self.calls = 0

    def retrieve(self, mentions):
        self.calls += 1
        return list(self._paths)

    def serialize_paths(self, paths):
        return [p.serialize() for p in paths]


def _path():
    return GraphPath(
        hops=[
            GraphFact(
                subject_key="K1", subject_name="Drug",
                predicate="treats",
                object_key="K2", object_name="Disease",
                confidence=0.9, polarity="positive",
                source_system="primekg", assertion_id="A1",
            )
        ],
        anchor_mention="Drug",
    )


def test_returns_text_and_graph_when_both_wired():
    text = _FakeText([_Result(_Chunk("text"), 1.0)])
    graph = _FakeGraphRetriever([_path()])
    linker = _FakeLinker([
        SimpleNamespace(mention_text="Drug", entity_id="K1", cui=None),
        SimpleNamespace(mention_text="Disease", entity_id="K2", cui=None),
    ])
    retriever = KGAugmentedRetriever(
        text_retriever=text, graph_retriever=graph, question_linker=linker
    )
    result = retriever.retrieve_with_kg("query")

    assert len(result.text_results) == 1
    assert len(result.graph_paths) == 1
    assert len(result.graph_path_lines) == 1
    assert "TREATS" in result.graph_path_lines[0]


def test_text_only_plan_when_both_wired_but_one_entity():
    text = _FakeText([_Result(_Chunk("text"), 1.0)])
    graph = _FakeGraphRetriever([_path()])
    linker = _FakeLinker([SimpleNamespace(mention_text="Drug", entity_id="K1", cui=None)])
    retriever = KGAugmentedRetriever(
        text_retriever=text, graph_retriever=graph, question_linker=linker
    )
    result = retriever.retrieve_with_kg("query")

    assert len(result.text_results) == 1
    assert result.graph_paths == []
    assert graph.calls == 0


def test_text_only_when_graph_components_missing():
    text = _FakeText([_Result(_Chunk("only text"), 1.0)])
    retriever = KGAugmentedRetriever(
        text_retriever=text, graph_retriever=None, question_linker=None
    )
    result = retriever.retrieve_with_kg("query")
    assert len(result.text_results) == 1
    assert result.graph_paths == []
    assert result.graph_path_lines == []


def test_graph_only_when_text_retriever_missing():
    graph = _FakeGraphRetriever([_path()])
    linker = _FakeLinker([SimpleNamespace(mention_text="X", entity_id="K1", cui=None)])
    retriever = KGAugmentedRetriever(
        text_retriever=None, graph_retriever=graph, question_linker=linker
    )
    result = retriever.retrieve_with_kg("query")
    assert result.text_results == []
    assert len(result.graph_paths) == 1


def test_compatibility_retrieve_method_returns_text_only():
    text = _FakeText([_Result(_Chunk("a"), 1.0), _Result(_Chunk("b"), 0.5)])
    graph = _FakeGraphRetriever([_path()])
    linker = _FakeLinker([SimpleNamespace(mention_text="X", entity_id="K1", cui=None)])
    retriever = KGAugmentedRetriever(
        text_retriever=text, graph_retriever=graph, question_linker=linker
    )
    out = retriever.retrieve("query")
    assert len(out) == 2  # text only — graph dropped through .retrieve()


def test_swallows_graph_errors_gracefully():
    text = _FakeText([_Result(_Chunk("safe"), 1.0)])

    class _Boom:
        def link_question(self, q):
            raise RuntimeError("scispacy crashed")

    graph = _FakeGraphRetriever([_path()])
    retriever = KGAugmentedRetriever(
        text_retriever=text, graph_retriever=graph, question_linker=_Boom()
    )
    result = retriever.retrieve_with_kg("query")
    # graph errored → empty graph paths, but text still works
    assert result.graph_paths == []
    assert len(result.text_results) == 1
