from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from src.layers.layer1_retrieval import PrimeKGExtractor
from src.schemas import QuestionEntity
from tools.grounding_ablation import audit_grounding_coverage, audit_grounding_records


class StaticLinker:
    backend_used = "rule_based"

    def __init__(self, entities):
        self._entities = list(entities)

    def link(self, text: str):
        del text
        return [entity.model_copy(deep=True) for entity in self._entities]


class SilentPubMedRetriever:
    backend_used = "disabled"

    def retrieve(self, question: str, k: int = 0):
        del question, k
        return []


class GroundingAwareRetriever:
    def __init__(self, *, primekg_path: Path, entities) -> None:
        self.config = SimpleNamespace(
            primekg_path=primekg_path,
            pubmed_cache_path=primekg_path / "pubmed_cache.jsonl",
            primekg_top_k=10,
            pubmed_top_k=0,
            use_relation_filter=False,
            relation_filter=None,
            relation_filter_overrides=None,
        )
        self.linker = StaticLinker(entities)
        self.kg_extractor = PrimeKGExtractor(primekg_path=primekg_path)
        self.pubmed = SilentPubMedRetriever()


def write_grounding_primekg_tables(path: Path) -> None:
    (path / "nodes.csv").write_text(
        "\n".join(
            [
                "node_index,node_id,node_type,node_name,node_source",
                "1,disease:h_pylori,disease,Helicobacter pylori infectious disease,primekg",
                "2,disease:gastritis,disease,gastritis,primekg",
            ]
        ),
        encoding="utf-8",
    )
    (path / "edges.csv").write_text(
        "\n".join(
            [
                "relation,display_relation,x_index,y_index",
                "disease_disease,associated disease,1,2",
            ]
        ),
        encoding="utf-8",
    )


def test_audit_grounding_records_runs_without_pubmed_or_trm(tmp_path: Path) -> None:
    write_grounding_primekg_tables(tmp_path)
    source = tmp_path / "medreason.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "row-1",
                        "question": "Most sensitive test for H pylori is-",
                        "answer": "Biopsy urease test",
                        "reasoning": "Helicobacter pylori infectious disease is associated with gastritis.",
                    }
                ),
                json.dumps(
                    {
                        "id": "row-2",
                        "question": "Typhoid investigation of choice in 1st week",
                        "answer": "Blood culture",
                        "reasoning": "Blood culture is often used early in typhoid fever.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    retriever = GroundingAwareRetriever(
        primekg_path=tmp_path,
        entities=[QuestionEntity(surface="H pylori", cui=None, primekg_node_id=None, entity_type="disease")],
    )

    payload = audit_grounding_records(
        source=source,
        split=None,
        grounding_mode="deterministic_v2",
        retriever=retriever,  # type: ignore[arg-type]
    )

    assert payload["summary"]["record_count"] == 2
    assert "entity_link_rate" in payload["summary"]
    assert "retrieved_edge_rate" in payload["summary"]
    assert "no_gold_skip_rate" in payload["summary"]


def test_audit_grounding_coverage_runs_without_graph_extraction(tmp_path: Path) -> None:
    write_grounding_primekg_tables(tmp_path)
    source = tmp_path / "medreason.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "row-1",
                        "question": "Most sensitive test for H pylori is-",
                        "answer": "Biopsy urease test",
                        "reasoning": "Helicobacter pylori infectious disease is associated with gastritis.",
                    }
                ),
                json.dumps(
                    {
                        "id": "row-2",
                        "question": "Deep transverse Perineus is related to which structure?",
                        "answer": "Unknown",
                        "reasoning": "An anatomy coverage gap sample.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = audit_grounding_coverage(
        source=source,
        split=None,
        grounding_mode="deterministic_v2",
        primekg_path=tmp_path,
    )

    assert payload["summary"]["record_count"] == 2
    assert "record_any_link_rate" in payload["summary"]
    assert "record_all_unresolved_rate" in payload["summary"]


def test_grounding_ablation_imports_without_layer1_for_coverage_path(tmp_path: Path) -> None:
    probe = tmp_path / "probe_import.py"
    repo_root = Path(__file__).resolve().parents[1]
    probe.write_text(
        "\n".join(
            [
                "import builtins",
                "import sys",
                f"sys.path.insert(0, {repr(str(repo_root))})",
                "real_import = builtins.__import__",
                "def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):",
                "    if name == 'src.layers.layer1_retrieval' or name.startswith('src.layers.layer1_retrieval'):",
                "        raise ImportError('blocked layer1 import')",
                "    return real_import(name, globals, locals, fromlist, level)",
                "builtins.__import__ = blocked_import",
                "import tools.grounding_ablation",
                "print('ok')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(probe)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
