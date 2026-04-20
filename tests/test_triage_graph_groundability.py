from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

import tools.triage_graph_groundability as triage


@dataclass(slots=True)
class FakeEdge:
    edge_id: str
    head: str
    tail: str
    relation: str
    display_relation: str
    source_reliability: float = 0.9
    amg_confidence: float = 1.0
    supporting_pmids: list[str] | None = None

    def __post_init__(self) -> None:
        if self.supporting_pmids is None:
            self.supporting_pmids = []


class FakeLinker:
    backend_used = "fake_linker"

    def link(self, text: str) -> list[triage.SimpleEntity]:
        lowered = text.lower()
        entities: list[triage.SimpleEntity] = []
        if "warfarin sodium" in lowered:
            entities.append(triage.SimpleEntity(surface="warfarin sodium", entity_type="drug"))
        elif "warfarin" in lowered:
            entities.append(triage.SimpleEntity(surface="warfarin", entity_type="drug"))
        if "ibuprofen" in lowered:
            entities.append(triage.SimpleEntity(surface="ibuprofen", entity_type="drug"))
        if "atrial fibrillation" in lowered:
            entities.append(triage.SimpleEntity(surface="atrial fibrillation", entity_type="disease"))
        return entities


class FakeKGExtractor:
    backend_used = "fake_graph"

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self.graph.add_node("drug:warfarin", name="warfarin", node_type="drug")
        self.graph.add_node("drug:ibuprofen", name="ibuprofen", node_type="drug")
        self.graph.add_node("disease:af", name="atrial fibrillation", node_type="disease")
        self.node_name_index = {
            "warfarin": {"drug:warfarin"},
            "ibuprofen": {"drug:ibuprofen"},
            "atrial fibrillation": {"disease:af"},
        }
        self.edges = [
            FakeEdge(
                edge_id="E_DRUG",
                head="drug:warfarin",
                tail="drug:ibuprofen",
                relation="drug_drug",
                display_relation="drug interaction",
            ),
            FakeEdge(
                edge_id="E_DISEASE",
                head="disease:af",
                tail="drug:warfarin",
                relation="indication",
                display_relation="indication",
            ),
        ]
        for edge in self.edges:
            self.graph.add_edge(edge.head, edge.tail, key=edge.edge_id, edge_id=edge.edge_id)
        self.node_degrees = {str(node_id): int(degree) for node_id, degree in self.graph.degree()}

    def resolve_seed_entities(self, seed_entities: list[triage.SimpleEntity]) -> list[triage.SimpleEntity]:
        resolved: list[triage.SimpleEntity] = []
        for entity in seed_entities:
            node_ids = self.node_name_index.get(triage._normalize_name(entity.surface), set())
            node_id = next(iter(node_ids)) if node_ids else None
            resolved.append(triage.SimpleEntity(entity.surface, entity.cui, node_id, entity.entity_type))
        return resolved

    def extract_2hop(self, seed_entities: list[triage.SimpleEntity], top_k: int = 200, relation_filter: set[str] | None = None) -> list[FakeEdge]:
        del top_k
        seed_ids = {entity.primekg_node_id for entity in seed_entities if entity.primekg_node_id}
        if not seed_ids:
            return []
        filtered: list[FakeEdge] = []
        for edge in self.edges:
            if relation_filter and edge.relation not in relation_filter:
                continue
            if edge.head in seed_ids or edge.tail in seed_ids:
                filtered.append(edge)
        return filtered

    def _iter_incident_edges(self, node_id: str, allowed_relations: set[str] | None = None) -> list[FakeEdge]:
        edges = []
        for edge in self.edges:
            if allowed_relations and edge.relation.lower() not in allowed_relations:
                continue
            if edge.head == node_id or edge.tail == node_id:
                edges.append(edge)
        return edges

    def _entity_matches_node_type(self, entity_type: str, node_type: str) -> bool:
        return entity_type == node_type or entity_type == "other"


class FakeEmbedder:
    def __call__(self, evidence: Any) -> dict[str, Any]:
        return {"inputs": [1, 2, 3], "evidence_edge_count": len(getattr(evidence, "subgraph_edges", []))}


def make_dependencies(*, warnings: list[str] | None = None, embedder: Any | None = None) -> triage.RuntimeDependencies:
    return triage.RuntimeDependencies(
        linker=FakeLinker(),
        kg_extractor=FakeKGExtractor(),
        embedder=FakeEmbedder() if embedder is None else embedder,
        warnings=warnings or [],
        default_top_k=10,
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def make_manifest_row(
    *,
    sample_id: str,
    row_index: int,
    bucket: str,
    error_count: int = 0,
    run_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "row_index": row_index,
        "question": f"Question {row_index}",
        "question_type": "factoid",
        "pass_success_count": 1 if bucket != "hard_absent" else 0,
        "support_rate": 1.0 if bucket != "hard_absent" else 0.0,
        "max_node_count": 1 if bucket != "hard_absent" else 0,
        "max_edge_count": 1 if bucket != "hard_absent" else 0,
        "max_labeled_token_count": 1 if bucket == "stable_graph" else 0,
        "bucket": bucket,
        "bucket_reason": "test",
        "passes": [],
        "dataset_name": "medmcqa",
        "id_in_dataset": row_index,
        "total_passes": 1,
        "error_count": error_count,
        "source": "data/medreason",
        "split": "train",
        "selected_passes": ["exact"],
        "fallback_warnings": [],
        "run_config": run_config
        or {
            "input_jsonl": "input.jsonl",
            "source": "data/medreason",
            "split": "train",
            "passes": ["exact"],
        },
        "sample_duration_ms": 12.5,
    }


def fake_map_medreason_edges(*, example: Any, evidence: Any, backend: str = "heuristic", selector: Any = None) -> tuple[list[str], None]:
    del example, backend, selector
    edges = list(getattr(evidence, "subgraph_edges", []))
    if not edges:
        return [], None
    return [edges[0].edge_id], None


def fake_path_to_token_sequence(gold_edge_ids: list[str], embedder_output: dict[str, Any], evidence: Any) -> tuple[list[int], dict[str, Any]]:
    del embedder_output, evidence
    return [0] * len(gold_edge_ids), {"labeled_token_count": len(gold_edge_ids)}


def test_load_records_respects_start_limit_and_dry_run(tmp_path: Path) -> None:
    input_path = tmp_path / "records.jsonl"
    write_jsonl(input_path, [{"question": f"Q{index}"} for index in range(6)])

    rows = list(triage.load_records(input_path, start_index=2, limit=3, dry_run=2))

    assert [row_index for row_index, _ in rows] == [2, 3]
    assert rows[0][1]["question"] == "Q2"
    assert triage.compute_target_total(input_path, start_index=2, limit=3, dry_run=2) == 2


def test_assign_bucket_rules_and_exception_override() -> None:
    assert triage.assign_bucket(
        pass_success_count=3,
        total_passes=3,
        support_rate=1.0,
        max_node_count=4,
        max_labeled_token_count=2,
        exception_override=None,
    ) == ("stable_graph", "all_passes_succeeded_with_labels")
    assert triage.assign_bucket(
        pass_success_count=2,
        total_passes=3,
        support_rate=2 / 3,
        max_node_count=3,
        max_labeled_token_count=0,
        exception_override=None,
    ) == ("threshold_a", "multi_pass_support_or_high_support_rate")
    assert triage.assign_bucket(
        pass_success_count=1,
        total_passes=3,
        support_rate=1 / 3,
        max_node_count=1,
        max_labeled_token_count=0,
        exception_override=None,
    ) == ("threshold_b", "single_pass_support_or_nodes_without_labels")
    assert triage.assign_bucket(
        pass_success_count=0,
        total_passes=3,
        support_rate=0.0,
        max_node_count=0,
        max_labeled_token_count=0,
        exception_override=None,
    ) == ("hard_absent", "no_graph_support_in_any_pass")
    assert triage.assign_bucket(
        pass_success_count=0,
        total_passes=3,
        support_rate=0.0,
        max_node_count=0,
        max_labeled_token_count=0,
        exception_override={"type": "RuntimeError", "message": "boom", "has_partial_graph": True},
    )[0] == "threshold_b"
    assert triage.assign_bucket(
        pass_success_count=0,
        total_passes=3,
        support_rate=0.0,
        max_node_count=0,
        max_labeled_token_count=0,
        exception_override={"type": "RuntimeError", "message": "boom", "has_partial_graph": False},
    )[0] == "hard_absent"


def test_reconcile_outputs_from_manifest_rebuilds_csv_and_buckets(tmp_path: Path) -> None:
    output_dir = tmp_path / "triage"
    output_dir.mkdir()
    manifest_rows = [
        make_manifest_row(sample_id="medmcqa:1", row_index=1, bucket="stable_graph"),
        make_manifest_row(sample_id="medmcqa:2", row_index=2, bucket="threshold_b", error_count=1),
    ]
    triage.write_jsonl_append(output_dir / "triage_manifest.jsonl", manifest_rows)
    (output_dir / "triage_manifest.csv").write_text("broken\n", encoding="utf-8")
    triage.write_jsonl_append(output_dir / "bucket_stable_graph.jsonl", manifest_rows[:1])

    state = triage.reconcile_outputs_from_manifest(
        output_dir=output_dir,
        started_at="2026-04-20T00:00:00+00:00",
        target_total=10,
        logger=logging.getLogger("triage-test"),
    )

    assert state.total_done == 2
    assert state.total_errors == 1
    assert state.skip_row_indexes == {1, 2}
    assert state.skip_sample_ids == {"medmcqa:1", "medmcqa:2"}

    with (output_dir / "triage_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 2

    stable_rows = [json.loads(line) for line in (output_dir / "bucket_stable_graph.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    threshold_b_rows = [json.loads(line) for line in (output_dir / "bucket_threshold_b.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(stable_rows) == 1
    assert len(threshold_b_rows) == 1


def test_validate_outputs_detects_duplicate_invalid_bucket_and_mismatch(tmp_path: Path) -> None:
    output_dir = tmp_path / "triage"
    output_dir.mkdir()
    manifest_rows = [
        make_manifest_row(sample_id="medmcqa:1", row_index=1, bucket="stable_graph"),
        make_manifest_row(sample_id="medmcqa:2", row_index=1, bucket="not_a_bucket"),
    ]
    triage.write_jsonl_append(output_dir / "triage_manifest.jsonl", manifest_rows)
    with (output_dir / "triage_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(triage.CSV_HEADERS))
        writer.writeheader()
        writer.writerow(triage.flatten_manifest_row(manifest_rows[0]))
    triage.write_jsonl_append(output_dir / "bucket_stable_graph.jsonl", [manifest_rows[0]])
    for bucket in ("threshold_a", "threshold_b", "hard_absent"):
        (output_dir / f"bucket_{bucket}.jsonl").write_text("", encoding="utf-8")
    (output_dir / "triage_summary.json").write_text(json.dumps({"bucket_counts": {"stable_graph": 0, "threshold_a": 0, "threshold_b": 0, "hard_absent": 0}}), encoding="utf-8")
    (output_dir / "progress.json").write_text(json.dumps({"bucket_counts": {"stable_graph": 0, "threshold_a": 0, "threshold_b": 0, "hard_absent": 0}}), encoding="utf-8")

    validation = triage.validate_outputs(output_dir)

    assert validation["valid"] is False
    assert any("Invalid bucket" in message for message in validation["messages"])
    assert any("Duplicate row_index" in message for message in validation["messages"])
    assert any("Manifest count" in message for message in validation["messages"])


def test_pass_execution_distinguishes_normalized_and_broad_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(triage, "map_medreason_edges", fake_map_medreason_edges)
    monkeypatch.setattr(triage, "path_to_token_sequence", fake_path_to_token_sequence)
    dependencies = make_dependencies()

    normalized_record = {
        "dataset_name": "medmcqa",
        "id_in_dataset": 11,
        "question": "Should AF be treated with anticoagulation?",
        "answer": "Yes",
        "reasoning": "Reasoning mentions [edge:E_DISEASE].",
    }
    broad_record = {
        "dataset_name": "medmcqa",
        "id_in_dataset": 12,
        "question": "Which anticoagulant is relevant here?",
        "options": "A. warfarin sodium\nB. saline",
        "answer": "A",
        "reasoning": "Reasoning mentions [edge:E_DRUG].",
    }

    exact_result = triage.run_pass_exact(row_index=11, record=normalized_record, dependencies=dependencies)
    normalized_result = triage.run_pass_normalized(row_index=11, record=normalized_record, dependencies=dependencies)
    broad_result = triage.run_pass_broad(row_index=12, record=broad_record, dependencies=dependencies)

    assert exact_result.success is False
    assert exact_result.node_count == 0
    assert normalized_result.success is True
    assert normalized_result.node_count > 0
    assert normalized_result.labeled_token_count == 1
    assert broad_result.success is True
    assert broad_result.linked_entity_count >= 1
    assert broad_result.candidate_path_count >= 1


def test_main_dry_run_and_validate_only_with_tensorization_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_path = tmp_path / "records.jsonl"
    output_dir = tmp_path / "triage"
    write_jsonl(
        input_path,
        [
            {
                "dataset_name": "medmcqa",
                "id_in_dataset": 21,
                "question": "Which anticoagulant is relevant here?",
                "options": "A. warfarin sodium\nB. saline",
                "answer": "A",
                "reasoning": "Reasoning mentions [edge:E_DRUG].",
            },
            {
                "dataset_name": "medmcqa",
                "id_in_dataset": 22,
                "question": "Another question",
                "answer": "B",
                "reasoning": "No graph.",
            },
        ],
    )

    def fake_build_runtime_dependencies(*, logger: logging.Logger) -> triage.RuntimeDependencies:
        del logger
        return make_dependencies(warnings=["tensorization_fallback:embedder_import_failed"], embedder=None)

    monkeypatch.setattr(triage, "build_runtime_dependencies", fake_build_runtime_dependencies)

    exit_code = triage.main(
        [
            "--input-jsonl",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--dry-run",
            "1",
            "--passes",
            "exact,broad",
        ]
    )

    assert exit_code == 0
    manifest_rows = [json.loads(line) for line in (output_dir / "triage_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(manifest_rows) == 1
    assert manifest_rows[0]["max_labeled_token_count"] == 0
    assert "tensorization_fallback:embedder_import_failed" in manifest_rows[0]["fallback_warnings"]

    validate_exit_code = triage.main(["--output-dir", str(output_dir), "--validate-only"])
    assert validate_exit_code == 0
