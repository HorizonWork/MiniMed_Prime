"""Render a multi-config ablation comparison.

``cli/evaluate.py`` runs `run_evaluation` once per config (text-only /
kg-only / kg+text) over the same loaded examples and hands the resulting
``BenchmarkReport``s here. Output is a single Markdown file (plus a
sibling JSON for downstream tooling).
"""

from __future__ import annotations

import datetime as _dt
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minimed_rag.benchmark.result_writer import BenchmarkReport


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1%}"


def _fmt_secs(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}s"


def _fmt_float(value: float | None, places: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{places}f}"


def _aggregate_per_dataset(reports: dict) -> dict[str, dict[str, dict]]:
    """Return ``{dataset_name: {config_name: stats_dict}}``."""
    per_ds: dict[str, dict[str, dict]] = defaultdict(dict)
    for config_name, report in reports.items():
        for ds_name, stats in report.per_dataset.items():
            per_ds[ds_name][config_name] = stats
    return per_ds


def write_ablation_report(
    reports: dict[str, BenchmarkReport],
    output_dir: str | Path,
    suite: str,
    *,
    timestamp: str | None = None,
) -> Path:
    """Write the comparison Markdown + JSON. Returns the Markdown path."""
    if not reports:
        raise ValueError("write_ablation_report requires at least one report")
    timestamp = timestamp or _dt.datetime.utcnow().isoformat()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_ts = timestamp.replace(":", "-").replace(".", "-")
    md_path = out_dir / f"ablation_{suite}_{safe_ts}.md"
    json_path = out_dir / f"ablation_{suite}_{safe_ts}.json"

    md_path.write_text(_render_markdown(suite, timestamp, reports), encoding="utf-8")
    json_path.write_text(_render_json(suite, timestamp, reports), encoding="utf-8")
    return md_path


def _render_markdown(suite: str, timestamp: str, reports: dict) -> str:
    lines: list[str] = [
        f"# Ablation Report: {suite}",
        "",
        f"- Timestamp: `{timestamp}`",
        f"- Configs evaluated: {len(reports)}",
        "",
        "## Overall",
        "",
        "| Config | Total | Accuracy | Invalid | Mean Latency | Retrieval Hit | KG Hit | Avg KG Paths |",
        "|--------|-------|----------|---------|--------------|---------------|--------|--------------|",
    ]
    for config_name, report in reports.items():
        diagnostics = report.retrieval_diagnostics
        retrieval_hit = _fmt_pct(diagnostics.corpus_hit_rate) if diagnostics else "-"
        kg_hit = (
            _fmt_pct(getattr(diagnostics, "kg_hit_rate", None))
            if diagnostics is not None
            else "-"
        )
        avg_kg = (
            _fmt_float(getattr(diagnostics, "avg_kg_paths", None), places=2)
            if diagnostics is not None
            else "-"
        )
        lines.append(
            f"| {config_name} "
            f"| {report.total} "
            f"| {_fmt_pct(report.accuracy)} "
            f"| {report.invalid} "
            f"| {_fmt_secs(report.mean_latency_s)} "
            f"| {retrieval_hit} "
            f"| {kg_hit} "
            f"| {avg_kg} |"
        )

    lines += ["", "## Per-Dataset Accuracy", ""]
    config_names = list(reports.keys())
    header = "| Dataset | " + " | ".join(config_names) + " |"
    sep = "|---------|" + "|".join(["---"] * len(config_names)) + "|"
    lines += [header, sep]
    per_ds = _aggregate_per_dataset(reports)
    for ds_name, by_config in per_ds.items():
        cells: list[str] = []
        for cfg in config_names:
            stats = by_config.get(cfg)
            if stats is None:
                cells.append("-")
            else:
                acc = stats.get("accuracy", 0.0)
                cells.append(_fmt_pct(acc))
        lines.append(f"| {ds_name} | " + " | ".join(cells) + " |")

    lines += ["", "## Config Hashes", ""]
    for config_name, report in reports.items():
        lines.append(f"- `{config_name}`: `{report.config_hash}`")

    lines.append("")
    return "\n".join(lines)


def _render_json(suite: str, timestamp: str, reports: dict) -> str:
    payload = {
        "suite": suite,
        "timestamp": timestamp,
        "configs": {},
    }
    for config_name, report in reports.items():
        diag = report.retrieval_diagnostics
        payload["configs"][config_name] = {
            "config_hash": report.config_hash,
            "total": report.total,
            "correct": report.correct,
            "invalid": report.invalid,
            "accuracy": report.accuracy,
            "invalid_rate": report.invalid_rate,
            "mean_latency_s": report.mean_latency_s,
            "per_dataset": report.per_dataset,
            "retrieval_diagnostics": asdict(diag) if diag is not None else None,
        }
    return json.dumps(payload, indent=2)


__all__ = ["write_ablation_report"]
