"""Unit tests for the Phase 6 ``generate-paths`` CLI."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from minimed_rag.cli import generate_paths as cli
from minimed_rag.domain.reasoning import ReasoningPath, ReasoningPathStep
from minimed_rag.nlp.linking.entity_linker import LinkedMention


runner = CliRunner()


def _mention(entity_id: str = "KG:Disease:cf", cui: str = "C0010674") -> LinkedMention:
    return LinkedMention(
        mention_text="cystic fibrosis",
        expanded_text="cystic fibrosis",
        entity_id=entity_id,
        concept_id=None,
        cui=cui,
        semantic_type="dsyn",
        confidence=0.95,
    )


def _path() -> ReasoningPath:
    return ReasoningPath(
        path_id="p1",
        task_id="t",
        path_type="positive",
        start_entity_id="KG:Disease:cf",
        end_entity_id="KG:Gene:cftr",
        task_type="gene_disease",
        metapath_template_id="gene_disease_direct",
        path_confidence=0.7,
        steps=[
            ReasoningPathStep(
                path_id="p1",
                step_index=0,
                subject_entity_id="KG:Disease:cf",
                predicate="gene_associated_with_disease",
                object_entity_id="KG:Gene:cftr",
                assertion_id="A1",
                edge_confidence=0.7,
                direction="forward",
            )
        ],
    )


class _StubLinker:
    def link_question(self, _text: str):
        return [_mention()]


class _StubFinder:
    def __init__(self, *_a, **_k) -> None:
        self.neo4j = SimpleNamespace(query=lambda *_a, **_k: [])

    def find_paths_from_anchor(self, *_a, **_k):
        return [_path()]

    def find_template_free_paths(self, *_a, **_k):
        return []


def test_generate_paths_writes_jsonl_with_header(tmp_path: Path) -> None:
    output = tmp_path / "paths.jsonl"

    settings_stub = SimpleNamespace(
        graph_version="kg_local",
        neo4j_uri="bolt://x",
        neo4j_user="u",
        neo4j_password="p",
        neo4j_database="neo4j",
        postgres_dsn="postgresql://x",
        scispacy_model="en_core_sci_lg",
        scispacy_umls_linker=True,
    )

    fake_neo4j = SimpleNamespace(close=lambda: None, query=lambda *_a, **_k: [])
    fake_pg = SimpleNamespace(close=lambda: None)

    class _MetapathRegistryStub:
        templates: dict = {}
        def get(self, template_id):
            raise KeyError(template_id)
        def get_for_task(self, _t):
            return []

    class _PredicateRegistryStub:
        rules: dict = {}
        def get(self, name):
            raise KeyError(name)
        def all(self):
            return {}
        def is_directional(self, _p):
            return True

    with patch.object(cli, "get_settings", return_value=settings_stub), \
         patch.object(cli, "neo4j_connect", return_value=fake_neo4j), \
         patch.object(cli, "postgres_connect", return_value=fake_pg), \
         patch.object(cli, "ScispacyEntityLinker", lambda **_kw: object()), \
         patch.object(cli, "SynonymFallbackLinker", lambda **_kw: object()), \
         patch.object(cli, "TermRepository", lambda **_kw: object()), \
         patch.object(cli, "GraphClient", lambda **_kw: object()), \
         patch.object(cli, "QuestionEntityLinker", lambda **_kw: _StubLinker()), \
         patch.object(cli, "MetapathRegistry", lambda *_a, **_kw: _MetapathRegistryStub()), \
         patch.object(cli, "PredicateRegistry", lambda *_a, **_kw: _PredicateRegistryStub()), \
         patch.object(cli, "PathFinder", _StubFinder):
        result = runner.invoke(
            cli.app,
            [
                "--question", "What gene is mutated in cystic fibrosis?",
                "--output", str(output),
            ],
        )

    assert result.exit_code == 0, result.output
    assert output.exists()

    lines = output.read_text().splitlines()
    assert len(lines) >= 2

    meta = json.loads(lines[0])
    assert "_meta" in meta
    assert meta["_meta"]["task_type"] == "gene_disease"
    assert meta["_meta"]["question"].startswith("What gene")
    assert meta["_meta"]["linked_entities"][0]["cui"] == "C0010674"

    record = json.loads(lines[1])
    assert record["metapath_template_id"] == "gene_disease_direct"
    assert record["steps"][0]["predicate"] == "gene_associated_with_disease"
