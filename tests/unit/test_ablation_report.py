"""Unit tests for the ablation comparison writer."""

from __future__ import annotations

import json

from minimed_rag.benchmark.ablation_report import write_ablation_report
from minimed_rag.benchmark.result_writer import BenchmarkReport, RetrievalDiagnostics


def _report(label: str, accuracy: float, *, kg=False) -> BenchmarkReport:
    diag = RetrievalDiagnostics(
        retriever=label,
        corpus_hit_rate=0.95,
        avg_score=0.42,
        avg_latency_s=0.12,
        kg_hit_rate=0.5 if kg else None,
        avg_kg_paths=8.0 if kg else None,
        avg_kg_latency_s=0.08 if kg else None,
        n_conflicts=3 if kg else None,
    )
    return BenchmarkReport(
        suite="mmlu",
        model="random",
        config={"label": label},
        config_hash="abc123def456",
        timestamp="2026-04-25T10:00:00",
        total=10,
        correct=int(accuracy * 10),
        invalid=0,
        accuracy=accuracy,
        invalid_rate=0.0,
        mean_latency_s=0.5,
        per_dataset={"mmlu_med": {"total": 10, "correct": int(accuracy * 10), "invalid": 0, "accuracy": accuracy}},
        results=[],
        retrieval_diagnostics=diag,
    )


def test_renders_three_config_table(tmp_path):
    reports = {
        "text_only": _report("text", 0.42),
        "kg_only": _report("kg", 0.38, kg=True),
        "kg_text": _report("kg+text", 0.47, kg=True),
    }
    md_path = write_ablation_report(reports, tmp_path, "mmlu", timestamp="2026-04-25T10:00:00")
    text = md_path.read_text()
    assert "# Ablation Report: mmlu" in text
    assert "text_only" in text
    assert "kg_only" in text
    assert "kg_text" in text
    assert "42.0%" in text
    assert "47.0%" in text
    # KG diagnostics should appear for kg-aware rows
    assert "50.0%" in text
    # config hashes section
    assert "abc123def456" in text


def test_writes_json_sibling(tmp_path):
    reports = {"text_only": _report("text", 0.5)}
    md_path = write_ablation_report(reports, tmp_path, "mmlu", timestamp="2026-04-25T10:00:00")
    json_path = md_path.with_suffix(".json")
    assert json_path.exists()
    payload = json.loads(json_path.read_text())
    assert payload["suite"] == "mmlu"
    assert "text_only" in payload["configs"]
    assert payload["configs"]["text_only"]["accuracy"] == 0.5


def test_raises_when_no_reports(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        write_ablation_report({}, tmp_path, "mmlu")


def test_per_dataset_section_columns_align(tmp_path):
    reports = {
        "a": _report("a", 0.4),
        "b": _report("b", 0.5),
    }
    md_path = write_ablation_report(reports, tmp_path, "mmlu", timestamp="2026-04-25T10:00:00")
    text = md_path.read_text()
    # Per-dataset table should have header with both config columns
    assert "## Per-Dataset Accuracy" in text
    # find header for per-dataset (after Per-Dataset header)
    pd_idx = text.index("## Per-Dataset Accuracy")
    pd_section = text[pd_idx:]
    assert "| Dataset | a | b |" in pd_section
