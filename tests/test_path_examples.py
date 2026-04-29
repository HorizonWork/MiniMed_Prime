"""Regression tests for docs/examples/path_*.json and evidence_bundle_*.json.

The path representation spec (docs/path_representation_spec.md) requires
each path's triples to satisfy the chain rule: triples[i+1].head equals
triples[i].tail. These tests catch the bug found in S1 review where
path_1/3/5 had been written as star-shaped subgraphs instead of chains.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kg.schema import load_schema

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "docs" / "examples"


def _path_files() -> list[Path]:
    return sorted(EXAMPLES_DIR.glob("path_*.json"))


def _bundle_files() -> list[Path]:
    return sorted(EXAMPLES_DIR.glob("evidence_bundle_*.json"))


@pytest.mark.parametrize("path_file", _path_files(), ids=lambda p: p.name)
def test_path_obeys_chain_rule(path_file: Path) -> None:
    with open(path_file) as f:
        data = json.load(f)
    triples = data["triples"]
    for i in range(len(triples) - 1):
        tail = triples[i][2]
        next_head = triples[i + 1][0]
        assert tail == next_head, (
            f"{path_file.name} triple {i+1}: head '{next_head}' "
            f"does not equal triple {i}'s tail '{tail}' (chain rule violated)"
        )


@pytest.mark.parametrize("path_file", _path_files(), ids=lambda p: p.name)
def test_path_relations_in_schema(path_file: Path) -> None:
    schema = load_schema()
    with open(path_file) as f:
        data = json.load(f)
    for h, r, t in data["triples"]:
        assert r in schema.edge_types, f"{path_file.name}: relation {r!r} not in schema"


@pytest.mark.parametrize("path_file", _path_files(), ids=lambda p: p.name)
def test_path_evidence_aligns_with_triples(path_file: Path) -> None:
    with open(path_file) as f:
        data = json.load(f)
    assert len(data["evidence"]) == len(data["triples"]), (
        f"{path_file.name}: evidence list length must equal triples length"
    )


@pytest.mark.parametrize("bundle_file", _bundle_files(), ids=lambda p: p.name)
def test_bundle_paths_each_obey_chain_rule(bundle_file: Path) -> None:
    with open(bundle_file) as f:
        data = json.load(f)
    for path in data["paths"]:
        triples = path["triples"]
        for i in range(len(triples) - 1):
            assert triples[i][2] == triples[i + 1][0], (
                f"{bundle_file.name} path {path['path_id']}: chain rule violated at hop {i+1}"
            )
