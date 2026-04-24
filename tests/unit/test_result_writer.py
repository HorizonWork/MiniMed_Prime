from __future__ import annotations

import json

from minimed_rag.benchmark.result_writer import BenchmarkReport, BenchmarkResult, write_results


def _result(**overrides) -> BenchmarkResult:
    defaults = {
        "example_id": "ex-1",
        "dataset": "tiny",
        "question": "Question?",
        "options": {"A": "Alpha", "B": "Beta"},
        "gold_answer": "A",
        "model_output": "A",
        "parsed_answer": "A",
        "correct": True,
        "latency_s": 0.01,
        "prompt": "Prompt",
    }
    defaults.update(overrides)
    return BenchmarkResult(**defaults)


def test_write_results_creates_jsonl_and_markdown(tmp_path):
    report = BenchmarkReport(
        suite="tiny-suite",
        model="stub-model",
        config={"limit": 2},
        config_hash="abc123",
        timestamp="2026-04-25T10:11:12.123456",
        total=2,
        correct=1,
        invalid=1,
        accuracy=1.0,
        invalid_rate=0.5,
        mean_latency_s=0.02,
        per_dataset={"tiny": {"total": 2, "correct": 1, "invalid": 1, "accuracy": 1.0}},
        results=[
            _result(),
            _result(
                example_id="ex-2",
                model_output="unclear",
                parsed_answer=None,
                correct=None,
                latency_s=0.03,
            ),
        ],
    )

    jsonl_path, md_path = write_results(report, tmp_path)

    assert jsonl_path.exists()
    assert md_path.exists()
    assert jsonl_path.name == "tiny-suite_2026-04-25T10-11-12-123456.jsonl"
    assert md_path.name == "tiny-suite_2026-04-25T10-11-12-123456.md"


def test_write_results_jsonl_contains_summary_then_results(tmp_path):
    report = BenchmarkReport(
        suite="tiny-suite",
        model="stub-model",
        config={},
        config_hash="abc123",
        timestamp="2026-04-25T10:11:12",
        total=1,
        correct=1,
        invalid=0,
        accuracy=1.0,
        invalid_rate=0.0,
        mean_latency_s=0.01,
        per_dataset={"tiny": {"total": 1, "correct": 1, "invalid": 0, "accuracy": 1.0}},
        results=[_result()],
    )

    jsonl_path, _ = write_results(report, tmp_path)
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]

    assert rows[0]["suite"] == "tiny-suite"
    assert "results" not in rows[0]
    assert rows[1]["example_id"] == "ex-1"


def test_write_results_markdown_contains_summary_table(tmp_path):
    report = BenchmarkReport(
        suite="tiny-suite",
        model="stub-model",
        config={},
        config_hash="abc123",
        timestamp="2026-04-25T10:11:12",
        total=1,
        correct=1,
        invalid=0,
        accuracy=1.0,
        invalid_rate=0.0,
        mean_latency_s=0.01,
        per_dataset={"tiny": {"total": 1, "correct": 1, "invalid": 0, "accuracy": 1.0}},
        results=[_result()],
    )

    _, md_path = write_results(report, tmp_path)
    markdown = md_path.read_text(encoding="utf-8")

    assert "# Benchmark Report: tiny-suite" in markdown
    assert "| tiny | 1 | 1 | 100.0% | 0 |" in markdown
