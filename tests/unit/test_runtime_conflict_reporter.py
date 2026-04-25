"""Unit tests for the heuristic runtime conflict reporter."""

from __future__ import annotations

import json
from dataclasses import dataclass

from minimed_rag.kg_build.runtime_conflict_reporter import (
    RuntimeConflictReporter,
    write_conflicts_jsonl,
)
from minimed_rag.retrieval.graph_retriever import GraphFact, GraphPath


@dataclass
class _Chunk:
    text: str
    id: str = "c1"
    source: str = "pubmed"


@dataclass
class _Result:
    chunk: _Chunk
    score: float = 0.0


def _fact(
    predicate: str,
    subj: str = "Drug X",
    obj: str = "Disease Y",
    polarity: str = "positive",
) -> GraphFact:
    return GraphFact(
        subject_key="KG:Drug:x",
        subject_name=subj,
        predicate=predicate,
        object_key="KG:Disease:y",
        object_name=obj,
        confidence=0.9,
        polarity=polarity,
        source_system="primekg",
        assertion_id="ASSERT:1",
    )


def _path(predicate: str = "treats", polarity: str = "positive") -> GraphPath:
    return GraphPath(hops=[_fact(predicate, polarity=polarity)], anchor_mention="Drug X")


def test_negation_in_text_flags_conflict():
    text = "Drug X does not treat Disease Y in any controlled trial."
    reporter = RuntimeConflictReporter()
    conflicts = reporter.find_conflicts(
        [_path("treats")],
        [_Result(chunk=_Chunk(text=text))],
    )
    assert len(conflicts) == 1
    assert conflicts[0].severity == "negation"
    assert "treat" in conflicts[0].rule


def test_polarity_flip_flags_conflict():
    text = "Drug X actually worsens Disease Y in some patients."
    reporter = RuntimeConflictReporter()
    conflicts = reporter.find_conflicts(
        [_path("treats")],
        [_Result(chunk=_Chunk(text=text))],
    )
    assert len(conflicts) == 1
    assert conflicts[0].severity == "polarity_flip"


def test_no_conflict_when_text_supports_assertion():
    text = "Drug X effectively treats Disease Y in clinical trials."
    reporter = RuntimeConflictReporter()
    conflicts = reporter.find_conflicts(
        [_path("treats")],
        [_Result(chunk=_Chunk(text=text))],
    )
    assert conflicts == []


def test_negative_graph_and_negated_text_is_not_conflict():
    text = "Drug X does not treat Disease Y in any controlled trial."
    reporter = RuntimeConflictReporter()
    conflicts = reporter.find_conflicts(
        [_path("treats", polarity="negative")],
        [_Result(chunk=_Chunk(text=text))],
    )
    assert conflicts == []


def test_negative_graph_and_affirmative_text_flags_conflict():
    text = "Drug X treats Disease Y in clinical trials."
    reporter = RuntimeConflictReporter()
    conflicts = reporter.find_conflicts(
        [_path("treats", polarity="negative")],
        [_Result(chunk=_Chunk(text=text))],
    )
    assert len(conflicts) == 1
    assert conflicts[0].rule.startswith("affirmation:")


def test_no_conflict_when_subject_and_object_far_apart():
    text = "Drug X is ancient. " + ("filler " * 100) + "Disease Y is modern. No relation discussed."
    reporter = RuntimeConflictReporter(co_occurrence_window=50)
    conflicts = reporter.find_conflicts(
        [_path("treats")],
        [_Result(chunk=_Chunk(text=text))],
    )
    assert conflicts == []


def test_empty_inputs_return_empty():
    reporter = RuntimeConflictReporter()
    assert reporter.find_conflicts([], []) == []
    assert reporter.find_conflicts([_path()], []) == []
    assert reporter.find_conflicts([], [_Result(chunk=_Chunk(text="x"))]) == []


def test_write_conflicts_jsonl_round_trips(tmp_path):
    reporter = RuntimeConflictReporter()
    text = "Drug X does not treat Disease Y."
    conflicts = reporter.find_conflicts(
        [_path("treats")],
        [_Result(chunk=_Chunk(text=text, id="cid", source="pubmed"))],
    )
    path = write_conflicts_jsonl(conflicts, tmp_path, "mmlu", "2026-04-25T10:00:00")
    assert path.exists()
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["fact_predicate"] == "treats"
    assert parsed["chunk_id"] == "cid"
    assert parsed["severity"] == "negation"
