"""Unit tests for the Phase 6 MetapathPlanner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from minimed_rag.reasoning.metapath_planner import MetapathPlanner


@dataclass(frozen=True, slots=True)
class _Step:
    subject_type: str
    predicate: str
    object_type: str


@dataclass(frozen=True, slots=True)
class _Template:
    name: str
    task_type: str
    pattern: tuple[_Step, ...]
    max_depth: int
    priority: float
    min_edge_confidence: float


class _FakeRegistry:
    def __init__(self, templates: Iterable[_Template]) -> None:
        self._templates = list(templates)

    def get_for_task(self, task_type: str) -> list[_Template]:
        return [t for t in self._templates if t.task_type == task_type]


def _t(name: str, task: str, prio: float) -> _Template:
    return _Template(
        name=name,
        task_type=task,
        pattern=(_Step("A", "p", "B"),),
        max_depth=1,
        priority=prio,
        min_edge_confidence=0.5,
    )


def test_returns_templates_sorted_priority_desc() -> None:
    reg = _FakeRegistry([_t("a", "tx", 0.5), _t("b", "tx", 0.9), _t("c", "tx", 0.7)])
    plan = MetapathPlanner(reg).plan_for_task("tx")
    assert [t.name for t in plan] == ["b", "c", "a"]


def test_filters_by_task_type() -> None:
    reg = _FakeRegistry(
        [_t("a", "tx", 0.5), _t("b", "other", 0.9), _t("c", "tx", 0.7)]
    )
    plan = MetapathPlanner(reg).plan_for_task("tx")
    assert {t.name for t in plan} == {"a", "c"}


def test_stable_tie_break_by_name() -> None:
    reg = _FakeRegistry(
        [_t("zeta", "tx", 0.7), _t("alpha", "tx", 0.7), _t("mu", "tx", 0.7)]
    )
    plan = MetapathPlanner(reg).plan_for_task("tx")
    assert [t.name for t in plan] == ["alpha", "mu", "zeta"]


def test_empty_when_no_templates_match() -> None:
    reg = _FakeRegistry([_t("a", "other", 0.9)])
    assert MetapathPlanner(reg).plan_for_task("tx") == []
