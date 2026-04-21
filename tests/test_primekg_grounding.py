from __future__ import annotations

from src.retrieval.primekg_grounding import node_lookup_keys, normalize_lookup_text, parse_node_cuis


def test_normalize_lookup_text_handles_surface_aliases_and_typos() -> None:
    assert normalize_lookup_text("H pylori") == "helicobacter pylori"
    assert normalize_lookup_text("Colle's fascia") == "colles fascia"
    assert normalize_lookup_text("common hepatic aery") == "common hepatic artery"


def test_node_lookup_keys_strip_generic_disease_suffixes() -> None:
    keys = node_lookup_keys("Helicobacter pylori infectious disease")

    assert "helicobacter pylori infectious disease" in keys
    assert "helicobacter pylori" in keys


def test_parse_node_cuis_splits_multiple_values() -> None:
    assert parse_node_cuis("C0019163| C0020538") == ["c0019163", "c0020538"]
