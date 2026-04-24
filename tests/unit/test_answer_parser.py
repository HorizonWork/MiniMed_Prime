from __future__ import annotations

from minimed_rag.benchmark.answer_parser import parse_answer, parse_mcq_answer, parse_yesno_answer


def test_parse_mcq_accepts_single_letter():
    assert parse_mcq_answer("b", {"A", "B", "C", "D"}) == "B"


def test_parse_mcq_extracts_explicit_answer():
    assert parse_mcq_answer("The answer is (C).", {"A", "B", "C", "D"}) == "C"


def test_parse_mcq_returns_none_for_invalid_option():
    assert parse_mcq_answer("Answer: E", {"A", "B", "C", "D"}) is None


def test_parse_yesno_extracts_biomedical_decision():
    assert parse_yesno_answer("Given the evidence, maybe.") == "maybe"


def test_parse_answer_routes_yesno_options():
    options = {"yes": "yes", "no": "no", "maybe": "maybe"}
    assert parse_answer("No.", options) == "no"


def test_parse_answer_routes_mcq_options():
    options = {"A": "alpha", "B": "beta"}
    assert parse_answer("Option B", options) == "B"
