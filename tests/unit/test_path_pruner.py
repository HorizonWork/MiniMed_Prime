"""Unit tests for the Phase 6 PathPruner."""

from __future__ import annotations

from minimed_rag.domain.reasoning import ReasoningPath, ReasoningPathStep
from minimed_rag.reasoning.path_pruner import PathPruner


def _step(s: str, p: str, o: str) -> ReasoningPathStep:
    return ReasoningPathStep(
        path_id="x",
        step_index=0,
        subject_entity_id=s,
        predicate=p,
        object_entity_id=o,
        assertion_id="A",
        edge_confidence=0.5,
    )


def _path(score: float, template: str, steps: list[ReasoningPathStep]) -> ReasoningPath:
    return ReasoningPath(
        path_id="p" + template + str(score),
        task_id="t",
        path_type="positive",
        start_entity_id=steps[0].subject_entity_id,
        end_entity_id=steps[-1].object_entity_id,
        task_type="tx",
        metapath_template_id=template,
        path_confidence=score,
        steps=steps,
    )


def test_min_score_floor_filters_low_confidence() -> None:
    paths = [
        _path(0.1, "t1", [_step("A", "p", "B")]),
        _path(0.04, "t1", [_step("A", "p", "B")]),  # below default 0.05
    ]
    kept = PathPruner(min_score=0.05).prune(paths)
    assert len(kept) == 1
    assert kept[0].path_confidence == 0.1


def test_top_k_caps_results() -> None:
    paths = [_path(c / 10, "t1", [_step("A", "p", str(c))]) for c in range(5, 0, -1)]
    kept = PathPruner(min_score=0.0, top_k=2).prune(paths)
    assert len(kept) == 2
    assert kept[0].path_confidence == 0.5
    assert kept[1].path_confidence == 0.4


def test_explicit_limit_overrides_top_k() -> None:
    paths = [_path(c / 10, "t1", [_step("A", "p", str(c))]) for c in (5, 4, 3)]
    kept = PathPruner(min_score=0.0, top_k=10).prune(paths, limit=1)
    assert len(kept) == 1


def test_dedupe_keys_on_predicate_tuple() -> None:
    duplicate_a = _path(0.9, "t1", [_step("A", "p", "B")])
    duplicate_b = _path(0.7, "t1", [_step("A", "p", "B")])  # same key
    distinct = _path(0.8, "t1", [_step("A", "q", "B")])
    deduped = PathPruner.dedupe([duplicate_a, duplicate_b, distinct])
    assert len(deduped) == 2
    # Original order preserved (first wins)
    assert deduped[0] is duplicate_a
    assert deduped[1] is distinct


def test_select_positive_paths_threshold() -> None:
    paths = [_path(0.6, "t", [_step("A", "p", "B")]),
             _path(0.4, "t", [_step("A", "p", "B")])]
    selected = PathPruner.select_positive_paths(paths, min_confidence=0.5)
    assert len(selected) == 1
    assert selected[0].path_confidence == 0.6
