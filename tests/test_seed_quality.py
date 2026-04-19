from __future__ import annotations

import json
from pathlib import Path

from src.training import classify_seed_record, filter_seed_jsonl


def _make_payload(
    *,
    gold_edge_ids: list[str],
    edge_mapper_backend: str,
    entity_linker_backend: str,
    primekg_backend: str,
    evidence_edge_count: int = 2,
) -> dict[str, object]:
    return {
        "gold_edge_ids": gold_edge_ids,
        "evidence": {
            "question_text": "Which antibiotic interacts with warfarin?",
            "question_type": "drug_interaction",
            "question_entities": [],
            "subgraph_edges": [{"edge_id": f"E_{index:05d}"} for index in range(evidence_edge_count)],
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
    assert decision.reasons == ["missing_gold_edge_ids"]


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

    goldish_rows = [json.loads(line) for line in (tmp_path / "quality" / "trm_seed_goldish.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    silver_rows = [json.loads(line) for line in (tmp_path / "quality" / "trm_seed_silver.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    reject_rows = [json.loads(line) for line in (tmp_path / "quality" / "trm_seed_rejects.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    summary_payload = json.loads((tmp_path / "quality" / "seed_quality_summary.json").read_text(encoding="utf-8"))

    assert goldish_rows[0]["quality"]["tier"] == "goldish"
    assert silver_rows[0]["quality"]["tier"] == "silver"
    assert reject_rows[0]["quality"]["tier"] == "reject"
    assert summary_payload["goldish_records"] == 1
    assert summary_payload["silver_records"] == 1
    assert summary_payload["rejected_records"] == 1
