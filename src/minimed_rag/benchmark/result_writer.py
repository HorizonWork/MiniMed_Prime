from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RetrievalDiagnostics:
    retriever: str
    corpus_hit_rate: float  # fraction of examples that got ≥1 chunk
    avg_score: float
    avg_latency_s: float
    # Phase 5 KG diagnostics — present when the retriever surfaces graph paths.
    kg_hit_rate: float | None = None      # fraction of examples with ≥1 graph path
    avg_kg_paths: float | None = None     # mean graph paths per example
    avg_kg_latency_s: float | None = None # mean graph retrieval latency
    n_conflicts: int | None = None        # graph-vs-text conflicts logged


@dataclass
class BenchmarkResult:
    example_id: str
    dataset: str
    question: str
    options: dict[str, str]
    gold_answer: str
    model_output: str
    parsed_answer: str | None
    correct: bool | None
    latency_s: float
    prompt: str
    retrieval_hit: bool = False
    retrieval_score: float = 0.0
    retrieval_latency_s: float = 0.0
    # Phase 5 KG diagnostics (defaults keep older serialised reports valid)
    kg_path_count: int = 0
    kg_latency_s: float = 0.0
    linked_entities: list[str] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    suite: str
    model: str
    config: dict
    config_hash: str
    timestamp: str
    total: int
    correct: int
    invalid: int
    accuracy: float
    invalid_rate: float
    mean_latency_s: float
    per_dataset: dict
    results: list[BenchmarkResult]
    retrieval_diagnostics: RetrievalDiagnostics | None = field(default=None)


def write_results(report: BenchmarkReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = report.timestamp.replace(":", "-").replace(".", "-")
    stem = f"{report.suite}_{ts}"

    jsonl_path = output_dir / f"{stem}.jsonl"
    _write_jsonl(report, jsonl_path)

    md_path = output_dir / f"{stem}.md"
    _write_markdown(report, md_path)

    return jsonl_path, md_path


def _write_jsonl(report: BenchmarkReport, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        summary = {k: v for k, v in asdict(report).items() if k != "results"}
        f.write(json.dumps(summary) + "\n")
        for result in report.results:
            f.write(json.dumps(asdict(result)) + "\n")


def _write_markdown(report: BenchmarkReport, path: Path) -> None:
    lines = [
        f"# Benchmark Report: {report.suite}",
        "",
        "## Summary",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Suite | `{report.suite}` |",
        f"| Model | `{report.model}` |",
        f"| Timestamp | {report.timestamp} |",
        f"| Config Hash | `{report.config_hash}` |",
        f"| Total | {report.total} |",
        f"| Correct | {report.correct} |",
        f"| Invalid | {report.invalid} |",
        f"| Accuracy | {report.accuracy:.1%} |",
        f"| Invalid Rate | {report.invalid_rate:.1%} |",
        f"| Mean Latency | {report.mean_latency_s:.3f}s |",
    ]

    if report.retrieval_diagnostics:
        d = report.retrieval_diagnostics
        lines += [
            "",
            "## Retrieval Diagnostics",
            "",
            "| Field | Value |",
            "|-------|-------|",
            f"| Retriever | `{d.retriever}` |",
            f"| Corpus Hit Rate | {d.corpus_hit_rate:.1%} |",
            f"| Avg Retrieval Score | {d.avg_score:.3f} |",
            f"| Avg Retrieval Latency | {d.avg_latency_s:.3f}s |",
        ]
        if d.kg_hit_rate is not None:
            lines += [
                f"| KG Hit Rate | {d.kg_hit_rate:.1%} |",
                f"| Avg KG Paths | {d.avg_kg_paths or 0.0:.2f} |",
                f"| Avg KG Latency | {d.avg_kg_latency_s or 0.0:.3f}s |",
            ]
        if d.n_conflicts is not None:
            lines.append(f"| Graph↔Text Conflicts | {d.n_conflicts} |")

    lines += [
        "",
        "## Per-Dataset Results",
        "",
        "| Dataset | Total | Correct | Accuracy | Invalid |",
        "|---------|-------|---------|----------|---------|",
    ]

    for ds_name, stats in report.per_dataset.items():
        acc = stats.get("accuracy", 0.0)
        lines.append(
            f"| {ds_name} | {stats['total']} | {stats['correct']} "
            f"| {acc:.1%} | {stats['invalid']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
