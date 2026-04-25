"""Unit tests for the Phase 5 QueryPlan decision."""

from __future__ import annotations

from minimed_rag.retrieval.query_planner import QueryPlan, QueryStrategy, decide_strategy


def test_strategy_text_only_when_no_entities_but_text_available():
    assert decide_strategy([], text_available=True, graph_available=True) == QueryPlan.TEXT_ONLY


def test_strategy_text_only_when_graph_unavailable():
    assert decide_strategy(["e1", "e2"], text_available=True, graph_available=False) == QueryPlan.TEXT_ONLY


def test_strategy_graph_only_when_text_unavailable_and_entities_present():
    assert decide_strategy(["e1"], text_available=False, graph_available=True) == QueryPlan.GRAPH_ONLY


def test_strategy_text_only_when_both_unavailable():
    assert decide_strategy(["e1", "e2"], text_available=False, graph_available=False) == QueryPlan.TEXT_ONLY


def test_strategy_hybrid_when_both_available_and_entities():
    assert decide_strategy(["e1", "e2"], text_available=True, graph_available=True) == QueryPlan.HYBRID


def test_strategy_text_only_below_threshold_with_one_entity():
    assert (
        decide_strategy(["e1"], text_available=True, graph_available=True, min_entities_for_kg=2)
        == QueryPlan.TEXT_ONLY
    )


def test_strategy_handles_none_input():
    assert decide_strategy(None, text_available=True, graph_available=True) == QueryPlan.TEXT_ONLY


def test_strategy_threshold_is_configurable():
    assert (
        decide_strategy(["a", "b", "c"], text_available=True, graph_available=True, min_entities_for_kg=5)
        == QueryPlan.TEXT_ONLY
    )


def test_query_strategy_alias_keeps_backcompat():
    assert QueryStrategy.HYBRID == QueryPlan.HYBRID
