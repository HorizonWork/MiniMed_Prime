"""Unit tests for the Phase 6 NegativeSampler."""

from __future__ import annotations

import random
from dataclasses import dataclass
from types import SimpleNamespace

from minimed_rag.domain.reasoning import ReasoningPath, ReasoningPathStep
from minimed_rag.reasoning.negative_sampler import NegativeSampler


def _step(s: str, p: str, o: str, direction: str = "forward") -> ReasoningPathStep:
    return ReasoningPathStep(
        path_id="x",
        step_index=0,
        subject_entity_id=s,
        predicate=p,
        object_entity_id=o,
        assertion_id="A",
        edge_confidence=0.7,
        direction=direction,
    )


def _path(steps: list[ReasoningPathStep]) -> ReasoningPath:
    return ReasoningPath(
        path_id="p1",
        task_id="t",
        path_type="positive",
        start_entity_id=steps[0].subject_entity_id,
        end_entity_id=steps[-1].object_entity_id,
        task_type="tx",
        metapath_template_id="m1",
        steps=steps,
    )


class _PredReg:
    def is_directional(self, pred: str) -> bool:
        return pred != "associated_with"

    def sample_incompatible_predicate(self, _s: str, _o: str, original: str) -> str:
        return "garbage_predicate"


class _Repo:
    def get_type(self, _entity_id: str) -> str:
        return "Drug"


@dataclass
class _FakeNeo4j:
    rows: list[dict]

    def query(self, _cypher: str, _params=None) -> list[dict]:
        return list(self.rows)


class _FakePathFinder:
    def __init__(self, neo4j) -> None:
        self.neo4j = neo4j

    def find_candidate_paths(self, *args, **kwargs):
        return []


def test_reversed_direction_path_negates_directional_steps() -> None:
    p = _path([_step("A", "treats", "B"), _step("B", "associated_with", "C")])
    sampler = NegativeSampler(
        entity_linker=None,
        path_finder=_FakePathFinder(_FakeNeo4j(rows=[])),
        predicate_registry=_PredReg(),
        entity_repo=_Repo(),
    )
    negs = sampler.generate_reversed_direction_paths([p])
    assert len(negs) == 1
    n = negs[0]
    # only the directional step is reversed; non-directional dropped
    assert len(n.steps) == 1
    assert n.steps[0].subject_entity_id == "B"
    assert n.steps[0].object_entity_id == "A"
    assert n.steps[0].direction == "reversed"
    assert n.validity_label == "invalid"
    assert n.negative_type == "reversed_causal_direction"


def test_relation_corruption_replaces_one_predicate() -> None:
    p = _path([_step("A", "treats", "B")])
    sampler = NegativeSampler(
        entity_linker=None,
        path_finder=_FakePathFinder(_FakeNeo4j(rows=[])),
        predicate_registry=_PredReg(),
        entity_repo=_Repo(),
        rng=random.Random(0),
    )
    negs = sampler.generate_relation_corruption_paths([p])
    assert len(negs) == 1
    assert negs[0].steps[0].predicate == "garbage_predicate"
    assert negs[0].negative_type == "relation_type_mismatch"
    assert negs[0].validity_label == "invalid"
    # original path untouched
    assert p.steps[0].predicate == "treats"


def test_random_walk_anchors_on_question_entities() -> None:
    rows = [
        {
            "assertion_id": "A1",
            "predicate": "treats",
            "confidence": 0.6,
            "neighbor_key": "N1",
        },
        {
            "assertion_id": "A2",
            "predicate": "targets",
            "confidence": 0.7,
            "neighbor_key": "N2",
        },
    ]
    finder = _FakePathFinder(_FakeNeo4j(rows=rows))
    sampler = NegativeSampler(
        entity_linker=None,
        path_finder=finder,
        predicate_registry=_PredReg(),
        entity_repo=_Repo(),
        num_random_walks=1,
        random_walk_length=1,
        rng=random.Random(0),
    )
    task = SimpleNamespace(
        task_id="t1",
        task_type="general",
        linked_question_entities=[SimpleNamespace(entity_id="X", cui=None)],
    )
    walks = sampler.generate_random_walk_paths(task, "kg_local")
    assert len(walks) == 1
    w = walks[0]
    assert w.path_type == "random_negative"
    assert w.validity_label == "invalid"
    assert w.negative_type == "random_walk"
    assert w.start_entity_id == "X"
    assert w.steps[0].subject_entity_id == "X"


def test_generate_skips_wrong_answer_when_no_options() -> None:
    sampler = NegativeSampler(
        entity_linker=SimpleNamespace(link_answer=lambda _x: []),
        path_finder=_FakePathFinder(_FakeNeo4j(rows=[])),
        predicate_registry=_PredReg(),
        entity_repo=_Repo(),
        num_random_walks=0,
    )
    task = SimpleNamespace(
        task_id="t",
        task_type="general",
        options=[],
        correct_answer=None,
        linked_question_entities=[],
    )
    out = sampler.generate(task, [], "kg_local")
    # all stubs return [], options empty, no positives → no negatives
    assert out == []


def test_entity_corruption_still_deferred() -> None:
    """entity_corruption is the only generator still gated on Phase 7
    infra (SapBERT semantic similarity); it must remain a no-op until
    that lands."""
    sampler = NegativeSampler(
        entity_linker=None,
        path_finder=_FakePathFinder(_FakeNeo4j(rows=[])),
        predicate_registry=_PredReg(),
        entity_repo=_Repo(),
    )
    assert sampler.generate_entity_corruption_paths([]) == []


def test_generic_hub_paths_returns_empty_without_repo() -> None:
    """Without an entity_stats_repo or hub_snapshot_id, generate_generic_hub_paths
    degrades to []. This is the path users hit before running
    `build-kg compute-graph-stats`."""
    sampler = NegativeSampler(
        entity_linker=None,
        path_finder=_FakePathFinder(_FakeNeo4j(rows=[])),
        predicate_registry=_PredReg(),
        entity_repo=_Repo(),
    )
    task = SimpleNamespace(linked_question_entities=[SimpleNamespace(entity_id="X")])
    assert sampler.generate_generic_hub_paths(task, "kg_local") == []


def test_generic_hub_paths_returns_empty_when_no_hubs_in_snapshot() -> None:
    class _EmptyStatsRepo:
        def get_top_degree_entities(self, snapshot_id, top_pct=0.01):
            return []

    sampler = NegativeSampler(
        entity_linker=None,
        path_finder=_FakePathFinder(_FakeNeo4j(rows=[])),
        predicate_registry=_PredReg(),
        entity_repo=_Repo(),
        entity_stats_repo=_EmptyStatsRepo(),
        hub_snapshot_id="snap",
    )
    task = SimpleNamespace(linked_question_entities=[SimpleNamespace(entity_id="X")])
    assert sampler.generate_generic_hub_paths(task, "kg_local") == []


def test_generic_hub_paths_anchored_returns_two_hop_negative() -> None:
    """With a stats repo that flags HUB1 as a top-degree entity, the
    fake Neo4j returns one canned 2-hop row; we expect a single negative
    path tagged generic_hub with 2 steps anchor → HUB1 → terminal."""
    rows = [
        {
            "a1_id": "A1",
            "a1_pred": "associated_with",
            "a1_conf": 0.4,
            "hub_key": "HUB1",
            "a2_id": "A2",
            "a2_pred": "associated_with",
            "a2_conf": 0.4,
            "terminal_key": "T1",
        }
    ]

    class _StatsRepo:
        def get_top_degree_entities(self, snapshot_id, top_pct=0.01):
            return ["HUB1"]

    sampler = NegativeSampler(
        entity_linker=None,
        path_finder=_FakePathFinder(_FakeNeo4j(rows=rows)),
        predicate_registry=_PredReg(),
        entity_repo=_Repo(),
        entity_stats_repo=_StatsRepo(),
        hub_snapshot_id="snap",
    )
    task = SimpleNamespace(
        task_id="t1",
        task_type="general",
        linked_question_entities=[SimpleNamespace(entity_id="X", cui=None)],
    )
    negs = sampler.generate_generic_hub_paths(task, "kg_local")
    assert len(negs) == 1
    n = negs[0]
    assert n.negative_type == "generic_hub"
    assert n.path_type == "random_negative"
    assert n.validity_label == "invalid"
    assert n.start_entity_id == "X"
    assert n.end_entity_id == "T1"
    assert [s.subject_entity_id for s in n.steps] == ["X", "HUB1"]
    assert [s.object_entity_id for s in n.steps] == ["HUB1", "T1"]
    assert [s.step_index for s in n.steps] == [0, 1]


def test_unsupported_paths_returns_empty_when_no_anchor() -> None:
    sampler = NegativeSampler(
        entity_linker=None,
        path_finder=_FakePathFinder(_FakeNeo4j(rows=[])),
        predicate_registry=_PredReg(),
        entity_repo=_Repo(),
    )
    task = SimpleNamespace(linked_question_entities=[])
    assert sampler.generate_unsupported_paths(task, "kg_local") == []


def test_unsupported_paths_one_hop_with_conflict_score_returns_negative() -> None:
    """Terminal-hop conflict_score above the default threshold (0.5) should
    yield an unsupported_kg_conflict negative with max_conflict_score set."""
    rows = [
        {
            "row": {
                "hops": 1,
                "a1_id": "A1",
                "a1_pred": "treats",
                "a1_conf": 0.7,
                "a1_score": 0.6,
                "hub_key": None,
                "a2_id": None,
                "a2_pred": None,
                "a2_conf": None,
                "a2_score": None,
                "terminal_key": "T1",
            }
        }
    ]
    sampler = NegativeSampler(
        entity_linker=None,
        path_finder=_FakePathFinder(_FakeNeo4j(rows=rows)),
        predicate_registry=_PredReg(),
        entity_repo=_Repo(),
    )
    task = SimpleNamespace(
        task_id="t1",
        task_type="treatment",
        linked_question_entities=[SimpleNamespace(entity_id="X", cui=None)],
    )
    negs = sampler.generate_unsupported_paths(task, "kg_local")
    assert len(negs) == 1
    n = negs[0]
    assert n.negative_type == "unsupported_kg_conflict"
    assert n.path_type == "hard_negative"
    assert n.validity_label == "invalid"
    assert n.max_conflict_score == 0.6
    assert n.start_entity_id == "X"
    assert n.end_entity_id == "T1"
    assert len(n.steps) == 1


def test_unsupported_paths_two_hop_uses_terminal_assertion_score() -> None:
    rows = [
        {
            "row": {
                "hops": 2,
                "a1_id": "A1",
                "a1_pred": "treats",
                "a1_conf": 0.6,
                "a1_score": 0.0,
                "hub_key": "M1",
                "a2_id": "A2",
                "a2_pred": "causes",
                "a2_conf": 0.5,
                "a2_score": 0.75,
                "terminal_key": "T1",
            }
        }
    ]
    sampler = NegativeSampler(
        entity_linker=None,
        path_finder=_FakePathFinder(_FakeNeo4j(rows=rows)),
        predicate_registry=_PredReg(),
        entity_repo=_Repo(),
    )
    task = SimpleNamespace(
        task_id="t1",
        task_type="treatment",
        linked_question_entities=[SimpleNamespace(entity_id="X", cui=None)],
    )
    negs = sampler.generate_unsupported_paths(task, "kg_local")
    assert len(negs) == 1
    n = negs[0]
    assert n.max_conflict_score == 0.75
    assert [s.subject_entity_id for s in n.steps] == ["X", "M1"]
    assert [s.object_entity_id for s in n.steps] == ["M1", "T1"]
