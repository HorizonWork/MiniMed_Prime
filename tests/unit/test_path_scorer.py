"""Unit tests for the Phase 6 PathScorer."""

from __future__ import annotations

from dataclasses import dataclass

from minimed_rag.domain.reasoning import ReasoningPath, ReasoningPathStep
from minimed_rag.reasoning.path_scorer import PathScorer


@dataclass
class _Template:
    name: str
    priority: float
    pattern: list = None  # unused in scorer


class _FakeRegistry:
    def __init__(self, templates: list[_Template]) -> None:
        self._t = {t.name: t for t in templates}

    def get(self, template_id: str) -> _Template:
        return self._t[template_id]


def _step(idx: int, conf: float, pred: str = "treats", direction: str = "forward") -> ReasoningPathStep:
    return ReasoningPathStep(
        path_id="p",
        step_index=idx,
        subject_entity_id=f"S{idx}",
        predicate=pred,
        object_entity_id=f"O{idx}",
        assertion_id=f"A{idx}",
        edge_confidence=conf,
        direction=direction,
    )


def _path(template_id: str, steps: list[ReasoningPathStep]) -> ReasoningPath:
    return ReasoningPath(
        path_id="p",
        task_id="t",
        path_type="positive",
        start_entity_id=steps[0].subject_entity_id,
        end_entity_id=steps[-1].object_entity_id,
        task_type="tx",
        metapath_template_id=template_id,
        steps=steps,
    )


def test_score_in_unit_range() -> None:
    reg = _FakeRegistry([_Template("t1", 1.0)])
    paths = [_path("t1", [_step(0, 0.9)])]
    PathScorer(reg).score(paths)
    assert 0.0 <= paths[0].path_confidence <= 1.0


def test_higher_min_conf_increases_score() -> None:
    reg = _FakeRegistry([_Template("t1", 1.0)])
    low = _path("t1", [_step(0, 0.3)])
    high = _path("t1", [_step(0, 0.9)])
    scorer = PathScorer(reg)
    scorer.score([low, high])
    assert high.path_confidence > low.path_confidence


def test_reversed_step_penalized() -> None:
    reg = _FakeRegistry([_Template("t1", 1.0)])
    forward = _path("t1", [_step(0, 0.9, direction="forward")])
    reversed_path = _path("t1", [_step(0, 0.9, direction="reversed")])
    PathScorer(reg).score([forward, reversed_path])
    assert reversed_path.path_confidence < forward.path_confidence


def test_template_inverse_not_penalized_like_reversed() -> None:
    reg = _FakeRegistry([_Template("t1", 1.0)])
    forward = _path("t1", [_step(0, 0.9, direction="forward")])
    inverse = _path("t1", [_step(0, 0.9, direction="template_inverse")])
    PathScorer(reg).score([forward, inverse])
    assert inverse.path_confidence == forward.path_confidence


def test_associated_with_penalty_applied() -> None:
    reg = _FakeRegistry([_Template("t1", 1.0)])
    plain = _path("t1", [_step(0, 0.9, pred="treats")])
    assoc = _path("t1", [_step(0, 0.9, pred="associated_with")])
    PathScorer(reg).score([plain, assoc])
    assert assoc.path_confidence < plain.path_confidence


def test_unknown_template_uses_default_prior() -> None:
    reg = _FakeRegistry([])
    p = _path("missing_template", [_step(0, 0.9)])
    PathScorer(reg).score([p])
    # Should still produce a finite, non-zero score using prior 0.5
    assert p.path_confidence > 0.0


def test_template_free_path_uses_fallback() -> None:
    reg = _FakeRegistry([])
    p = _path("", [_step(0, 0.9)])
    PathScorer(reg).score([p])
    assert p.path_confidence > 0.0


def test_longer_paths_decay_score() -> None:
    reg = _FakeRegistry([_Template("t1", 1.0)])
    short = _path("t1", [_step(0, 0.9)])
    long = _path(
        "t1",
        [
            _step(0, 0.9),
            _step(1, 0.9),
            _step(2, 0.9),
        ],
    )
    PathScorer(reg).score([short, long])
    assert long.path_confidence < short.path_confidence


# ---- hub_penalty tests ---------------------------------------------------


class _FakeStatsRepo:
    """Returns canned degree counts; ``p50`` configurable."""

    def __init__(self, degree_by_id: dict[str, int], p50: float = 10.0) -> None:
        self._degree = degree_by_id
        self._p50 = p50

    def get_degree(self, entity_id: str, snapshot_id: str) -> int | None:
        return self._degree.get(entity_id)

    def get_p50_degree(self, snapshot_id: str) -> float:
        return self._p50

    def get_top_degree_entities(self, snapshot_id: str, top_pct: float = 0.01) -> list[str]:
        return list(self._degree)


def test_hub_penalty_no_op_without_repo() -> None:
    """Backward-compat: with no entity_stats_repo, hub_penalty = 1.0 and the
    score matches the legacy formula."""
    reg = _FakeRegistry([_Template("t1", 1.0)])
    p_no_stats = _path("t1", [_step(0, 0.9), _step(1, 0.9)])
    p_with_repo = _path("t1", [_step(0, 0.9), _step(1, 0.9)])
    PathScorer(reg).score([p_no_stats])
    PathScorer(reg, entity_stats_repo=None).score([p_with_repo])
    assert p_no_stats.path_confidence == p_with_repo.path_confidence


def test_hub_penalty_demotes_path_through_high_degree_entity() -> None:
    """A 2-hop path whose intermediate entity is a hub (degree=200, p50=10)
    should score lower than the same path with a non-hub intermediate."""
    reg = _FakeRegistry([_Template("t1", 1.0)])
    # In a 2-hop path, intermediate = step[0].object_entity_id = "O0"
    # _step(0, ...) has subject="S0", object="O0"; _step(1, ...) has "S1", "O1".
    hub_repo = _FakeStatsRepo({"O0": 200}, p50=10.0)
    nohub_repo = _FakeStatsRepo({"O0": 5}, p50=10.0)
    hub_path = _path("t1", [_step(0, 0.9), _step(1, 0.9)])
    nohub_path = _path("t1", [_step(0, 0.9), _step(1, 0.9)])
    PathScorer(reg, entity_stats_repo=hub_repo, hub_snapshot_id="snap").score([hub_path])
    PathScorer(reg, entity_stats_repo=nohub_repo, hub_snapshot_id="snap").score([nohub_path])
    assert hub_path.path_confidence < nohub_path.path_confidence


def test_hub_penalty_skips_endpoints() -> None:
    """Endpoints (anchor + terminal) should NOT be penalized for high
    degree — only intermediates. A 1-hop path has no intermediates, so
    hub_penalty must be 1.0 even with a "hub" subject/object."""
    reg = _FakeRegistry([_Template("t1", 1.0)])
    repo = _FakeStatsRepo({"S0": 9999, "O0": 9999}, p50=10.0)
    p_with_hub_endpoints = _path("t1", [_step(0, 0.9)])
    p_without_repo = _path("t1", [_step(0, 0.9)])
    PathScorer(reg, entity_stats_repo=repo, hub_snapshot_id="snap").score(
        [p_with_hub_endpoints]
    )
    PathScorer(reg).score([p_without_repo])
    assert p_with_hub_endpoints.path_confidence == p_without_repo.path_confidence


def test_hub_penalty_no_op_when_p50_degenerate() -> None:
    """If the snapshot is empty (p50 returned as 0.0), hub_penalty falls
    back to 1.0 — we can't calibrate without a baseline."""
    reg = _FakeRegistry([_Template("t1", 1.0)])
    repo = _FakeStatsRepo({"O0": 200}, p50=0.0)
    p = _path("t1", [_step(0, 0.9), _step(1, 0.9)])
    p_no_repo = _path("t1", [_step(0, 0.9), _step(1, 0.9)])
    PathScorer(reg, entity_stats_repo=repo, hub_snapshot_id="snap").score([p])
    PathScorer(reg).score([p_no_repo])
    assert p.path_confidence == p_no_repo.path_confidence


def test_hub_penalty_skips_unknown_entities() -> None:
    """A missing degree lookup is treated as "we don't know" rather than
    "degree 0" — the entity is just skipped."""
    reg = _FakeRegistry([_Template("t1", 1.0)])
    repo = _FakeStatsRepo({}, p50=10.0)  # nothing known
    p = _path("t1", [_step(0, 0.9), _step(1, 0.9)])
    p_no_repo = _path("t1", [_step(0, 0.9), _step(1, 0.9)])
    PathScorer(reg, entity_stats_repo=repo, hub_snapshot_id="snap").score([p])
    PathScorer(reg).score([p_no_repo])
    assert p.path_confidence == p_no_repo.path_confidence
