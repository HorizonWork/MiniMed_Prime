"""Unit tests for ContextBuilder.build_hybrid_context_text."""

from __future__ import annotations

from dataclasses import dataclass

from minimed_rag.retrieval.context_builder import ContextBuilder


@dataclass
class _Chunk:
    text: str


@dataclass
class _Result:
    chunk: _Chunk
    score: float = 0.0


def _r(text: str) -> _Result:
    return _Result(chunk=_Chunk(text=text))


def test_emits_both_sections_when_present():
    out = ContextBuilder.build_hybrid_context_text(
        ["A --[treats]--> B", "C --[causes]--> D"],
        [_r("Some textbook excerpt about A and B.")],
    )
    assert "[Graph Evidence]" in out
    assert "[Text Evidence]" in out
    assert "A --[treats]--> B" in out
    assert "Some textbook excerpt" in out
    # graph block precedes text block
    assert out.index("[Graph Evidence]") < out.index("[Text Evidence]")


def test_omits_graph_section_when_no_paths():
    out = ContextBuilder.build_hybrid_context_text([], [_r("only text here")])
    assert "[Graph Evidence]" not in out
    assert "[Text Evidence]" in out
    assert "only text here" in out


def test_omits_text_section_when_no_chunks():
    out = ContextBuilder.build_hybrid_context_text(["A --> B"], [])
    assert "[Graph Evidence]" in out
    assert "[Text Evidence]" not in out


def test_returns_empty_when_neither_side_present():
    assert ContextBuilder.build_hybrid_context_text([], []) == ""


def test_strips_blank_graph_lines():
    out = ContextBuilder.build_hybrid_context_text(["", "  ", "real line"], [])
    assert "real line" in out
    assert "- real line" not in out


def test_max_chars_budget_is_respected():
    long_text = "X" * 5000
    out = ContextBuilder.build_hybrid_context_text(
        ["A --> B"],
        [_r(long_text)],
        max_chars=500,
    )
    assert len(out) <= 600  # allow small budget overshoot from headers/separators


def test_invalid_ratio_raises():
    import pytest
    with pytest.raises(ValueError):
        ContextBuilder.build_hybrid_context_text([], [], graph_budget_ratio=1.5)


def test_graph_budget_ratio_caps_graph_block_size():
    huge_lines = [f"path {i} --> end" for i in range(200)]
    out = ContextBuilder.build_hybrid_context_text(
        huge_lines,
        [_r("text payload " * 200)],
        max_chars=2000,
        graph_budget_ratio=0.4,
    )
    # graph block should not exceed roughly 0.4 * 2000 = 800 chars
    graph_section_end = out.index("[Text Evidence]") if "[Text Evidence]" in out else len(out)
    graph_section = out[:graph_section_end]
    assert len(graph_section) <= 1000
