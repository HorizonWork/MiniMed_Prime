from __future__ import annotations

import json
from pathlib import Path

from src.training import classify_seed_record, compute_seed_coverage, filter_seed_jsonl


def _make_payload(
    *,
    gold_edge_ids: list[str],
    edge_mapper_backend: str,
    entity_linker_backend: str,
    primekg_backend: str,
    evidence_edge_count: int = 2,
    evidence_edge_ids: list[str] | None = None,
    direct_edge_ids: list[str] | None = None,
    question_seed_node: str = "node:0",
    extra_edges: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    edge_ids = evidence_edge_ids if evidence_edge_ids is not None else list(gold_edge_ids[:evidence_edge_count])
    while len(edge_ids) < evidence_edge_count:
        edge_ids.append(f"E_EXTRA_{len(edge_ids):05d}")
    subgraph_edges = [
        {
            "edge_id": edge_id,
            "head": f"node:{index}",
            "tail": f"node:{index + 1}",
            "relation": "drug_drug",
            "display_relation": "interacts with",
            "source_reliability": 0.95,
            "amg_confidence": 1.0,
            "supporting_pmids": [],
        }
        for index, edge_id in enumerate(edge_ids)
    ]
    if extra_edges:
        subgraph_edges.extend(extra_edges)
    return {
        "gold_edge_ids": gold_edge_ids,
        "evidence": {
            "question_text": "Which antibiotic interacts with warfarin?",
            "question_type": "drug_interaction",
            "question_entities": [
                {
                    "surface": "warfarin",
                    "cui": "C0043031",
                    "primekg_node_id": question_seed_node,
                    "entity_type": "drug",
                }
            ],
            "subgraph_edges": subgraph_edges,
            "pubmed_passages": [],
            "metadata": {
                "entity_linker_backend": entity_linker_backend,
                "backend_used": {
                    "entity_linker": entity_linker_backend,
                    "primekg": primekg_backend,
                    "pubmed": "bm25+medcpt",
                },
            },
        },
        "metadata": {
            "edge_mapping": {
                "backend_used": edge_mapper_backend,
                "direct_edge_ids": direct_edge_ids if direct_edge_ids is not None else gold_edge_ids,
                "selected_edge_ids": gold_edge_ids,
            }
        },
    }


def test_classify_seed_record_marks_llm_and_scispacy_run_as_goldish() -> None:
    decision = classify_seed_record(
        _make_payload(
            gold_edge_ids=["E_00001", "E_00002"],
            edge_mapper_backend="llm_openai",
            entity_linker_backend="scispacy_umls",
            primekg_backend="networkx_edge_node_csv",
        )
    )

    assert decision.tier == "goldish"
    assert decision.reasons == ["passed_goldish_gate"]
    assert decision.gold_edge_coverage == 1.0
    assert decision.gold_node_coverage == 1.0
    assert decision.labeled_token_ratio == 1.0
    assert decision.labeled_token_count == 3


def test_classify_seed_record_demotes_rule_based_run_to_silver() -> None:
    decision = classify_seed_record(
        _make_payload(
            gold_edge_ids=["E_00001"],
            edge_mapper_backend="heuristic",
            entity_linker_backend="rule_based",
            primekg_backend="networkx_csv",
        )
    )

    assert decision.tier == "silver"
    assert "edge_mapper_backend:heuristic" in decision.reasons
    assert "entity_linker_backend:rule_based" in decision.reasons


def test_classify_seed_record_rejects_missing_gold_edges() -> None:
    decision = classify_seed_record(
        _make_payload(
            gold_edge_ids=[],
            edge_mapper_backend="direct",
            entity_linker_backend="scispacy_umls",
            primekg_backend="networkx_csv",
        )
    )

    assert decision.tier == "reject"
    assert decision.reasons == ["empty_gold_path"]
    assert decision.labeled_token_count == 0


def test_seed_coverage_reports_full_coverage() -> None:
    coverage = compute_seed_coverage(
        _make_payload(
            gold_edge_ids=["E_00001", "E_00002"],
            edge_mapper_backend="direct",
            entity_linker_backend="scispacy_umls",
            primekg_backend="networkx_csv",
        )
    )

    assert coverage.gold_edge_coverage == 1.0
    assert coverage.gold_node_coverage == 1.0
    assert coverage.labeled_token_ratio == 1.0
    assert coverage.labeled_token_count == 3
    assert coverage.missing_gold_edges == []
    assert coverage.ppr_pruned_gold_nodes == []


def test_classify_seed_record_demotes_partial_coverage_to_silver() -> None:
    decision = classify_seed_record(
        _make_payload(
            gold_edge_ids=["E_00001"],
            direct_edge_ids=["E_00001", "E_00002"],
            edge_mapper_backend="direct",
            entity_linker_backend="scispacy_umls",
            primekg_backend="networkx_csv",
            evidence_edge_ids=["E_00001"],
        )
    )

    assert decision.tier == "silver"
    assert decision.gold_edge_coverage == 0.5
    assert "missing_edge_mapping" in decision.reasons
    assert "gold_edge_coverage_below_threshold" in decision.reasons


def test_classify_seed_record_rejects_zero_coverage() -> None:
    decision = classify_seed_record(
        _make_payload(
            gold_edge_ids=["E_99999"],
            edge_mapper_backend="direct",
            entity_linker_backend="scispacy_umls",
            primekg_backend="networkx_csv",
            evidence_edge_ids=["E_00001"],
        )
    )

    assert decision.tier == "reject"
    assert decision.gold_edge_coverage == 0.0
    assert decision.labeled_token_count == 0
    assert "missing_edge_mapping" in decision.reasons


def test_classify_seed_record_rejects_gold_path_cut_by_top_k() -> None:
    payload = _make_payload(
        gold_edge_ids=["E_GOLD"],
        edge_mapper_backend="direct",
        entity_linker_backend="scispacy_umls",
        primekg_backend="networkx_csv",
        evidence_edge_ids=[],
        evidence_edge_count=0,
        question_seed_node="seed",
        extra_edges=[
            {
                "edge_id": "E_SEED",
                "head": "seed",
                "tail": "helper",
                "relation": "drug_drug",
                "display_relation": "interacts with",
                "source_reliability": 0.95,
                "amg_confidence": 1.0,
                "supporting_pmids": [],
            },
            {
                "edge_id": "E_GOLD",
                "head": "gold:a",
                "tail": "gold:b",
                "relation": "drug_drug",
                "display_relation": "interacts with",
                "source_reliability": 0.95,
                "amg_confidence": 1.0,
                "supporting_pmids": [],
            },
        ],
    )

    decision = classify_seed_record(payload, coverage_top_k=1)

    assert decision.tier == "reject"
    assert decision.gold_edge_coverage == 1.0
    assert decision.gold_node_coverage == 1.0
    assert decision.labeled_token_count == 0
    assert "missing_due_to_ppr_pruning" in decision.reasons


def test_filter_seed_jsonl_splits_outputs_and_writes_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "trm_seed.jsonl"
    rows = [
        _make_payload(
            gold_edge_ids=["E_00001", "E_00002"],
            edge_mapper_backend="llm_openai",
            entity_linker_backend="scispacy_umls",
            primekg_backend="networkx_edge_node_csv",
        ),
        _make_payload(
            gold_edge_ids=["E_00003"],
            edge_mapper_backend="heuristic",
            entity_linker_backend="rule_based",
            primekg_backend="networkx_csv",
        ),
        _make_payload(
            gold_edge_ids=[],
            edge_mapper_backend="direct",
            entity_linker_backend="scispacy_umls",
            primekg_backend="networkx_csv",
        ),
    ]
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = filter_seed_jsonl(input_jsonl=input_path, output_dir=tmp_path / "quality")

    assert summary.goldish_records == 1
    assert summary.silver_records == 1
    assert summary.rejected_records == 1
    assert summary.accepted_goldish == 1
    assert summary.demoted_silver == 1
    assert summary.rejected == 1

    goldish_rows = [json.loads(line) for line in (tmp_path / "quality" / "trm_seed_goldish.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    silver_rows = [json.loads(line) for line in (tmp_path / "quality" / "trm_seed_silver.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    reject_rows = [json.loads(line) for line in (tmp_path / "quality" / "trm_seed_rejects.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    summary_payload = json.loads((tmp_path / "quality" / "seed_quality_summary.json").read_text(encoding="utf-8"))

    assert goldish_rows[0]["quality"]["tier"] == "goldish"
    assert goldish_rows[0]["quality"]["labeled_token_count"] > 0
    assert silver_rows[0]["quality"]["tier"] == "silver"
    assert reject_rows[0]["quality"]["tier"] == "reject"
    assert summary_payload["goldish_records"] == 1
    assert summary_payload["silver_records"] == 1
    assert summary_payload["rejected_records"] == 1
    assert summary_payload["coverage_metric_means"]["goldish"]["labeled_token_ratio"] == 1.0
    assert summary_payload["top_failure_reasons"]
