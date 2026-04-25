from __future__ import annotations

import datetime
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .answer_parser import parse_answer
from .datasets.base import MCQExample
from .prompt_templates import format_prompt
from .providers.base import ModelProvider
from .result_writer import BenchmarkReport, BenchmarkResult, RetrievalDiagnostics, write_results

if TYPE_CHECKING:
    from minimed_rag.retrieval.bm25_retriever import RetrievalResult
    from minimed_rag.retrieval.kg_augmented_retriever import KGAugmentedRetriever


class ChunkRetriever(Protocol):
    def retrieve(self, query: str, k: int | None = None) -> list[RetrievalResult]: ...


def _config_hash(config: dict) -> str:
    canon = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


def _is_kg_retriever(retriever: Any) -> bool:
    return retriever is not None and hasattr(retriever, "retrieve_with_kg")


def run_evaluation(
    suite: str,
    examples: list[MCQExample],
    provider: ModelProvider,
    output_dir: str | Path = "artifacts/reports",
    *,
    retriever: ChunkRetriever | None = None,
    kg_retriever: KGAugmentedRetriever | None = None,
    conflict_reporter: Any = None,
    conflict_output_dir: str | Path | None = None,
    config_label: str | None = None,
    max_context_chars: int = 2500,
    graph_budget_ratio: float = 0.4,
    verbose: bool = False,
) -> BenchmarkReport:
    """Run a benchmark slice.

    Phase 5 additions:
    - ``kg_retriever`` (KGAugmentedRetriever): when supplied, calls
      ``retrieve_with_kg`` and merges graph paths + text chunks via
      ``ContextBuilder.build_hybrid_context_text`` before prompting.
      Plain ``retriever`` is ignored when ``kg_retriever`` is set.
    - ``conflict_reporter``: optional ``RuntimeConflictReporter`` — runs
      per-example against the graph paths and text chunks; conflicts are
      flushed to a JSONL alongside the main report.
    - ``config_label``: label embedded in the config for ablation reports.
    """
    active_retriever: Any = kg_retriever or retriever
    retriever_name = (
        getattr(active_retriever, "name", None)
        or (type(active_retriever).__name__ if active_retriever is not None else "none")
    )
    kg_active = _is_kg_retriever(active_retriever)
    config = {
        **provider.config,
        "suite": suite,
        "n_examples": len(examples),
        "retriever": retriever_name,
        "kg_active": kg_active,
        "config_label": config_label or retriever_name,
        "max_context_chars": max_context_chars,
        "graph_budget_ratio": graph_budget_ratio,
    }
    c_hash = _config_hash(config)
    timestamp = datetime.datetime.utcnow().isoformat()

    results: list[BenchmarkResult] = []
    per_dataset: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0, "invalid": 0})

    retrieval_hits = 0
    retrieval_scores: list[float] = []
    retrieval_latencies: list[float] = []
    kg_path_counts: list[int] = []
    kg_latencies: list[float] = []
    kg_hits = 0
    all_conflicts: list = []

    for idx, ex in enumerate(examples):
        context_text = ""
        hit = False
        r_score = 0.0
        r_latency = 0.0
        kg_paths_count = 0
        kg_latency = 0.0
        linked_entity_names: list[str] = []

        if kg_active:
            kg_result = active_retriever.retrieve_with_kg(ex.question)
            text_results = kg_result.text_results
            graph_lines = kg_result.graph_path_lines
            r_latency = kg_result.text_latency_s
            kg_latency = kg_result.graph_latency_s
            kg_paths_count = len(kg_result.graph_paths)
            linked_entity_names = [
                getattr(m, "mention_text", "") or "" for m in kg_result.linked_mentions
            ]

            if text_results:
                hit = True
                r_score = text_results[0].score
            from minimed_rag.retrieval.context_builder import ContextBuilder

            context_text = ContextBuilder.build_hybrid_context_text(
                graph_lines,
                text_results,
                max_chars=max_context_chars,
                graph_budget_ratio=graph_budget_ratio,
            )

            retrieval_latencies.append(r_latency)
            retrieval_scores.append(r_score)
            kg_path_counts.append(kg_paths_count)
            kg_latencies.append(kg_latency)
            if hit:
                retrieval_hits += 1
            if kg_paths_count > 0:
                kg_hits += 1

            if conflict_reporter is not None and kg_result.graph_paths:
                conflicts = conflict_reporter.find_conflicts(
                    kg_result.graph_paths, text_results
                )
                all_conflicts.extend(conflicts)

        elif retriever is not None:
            t_r0 = time.perf_counter()
            retrieved = retriever.retrieve(ex.question)
            r_latency = time.perf_counter() - t_r0

            if retrieved:
                hit = True
                r_score = retrieved[0].score
                from minimed_rag.retrieval.context_builder import ContextBuilder

                context_text = ContextBuilder.build_context_text(
                    retrieved,
                    max_chars=max_context_chars,
                )

            retrieval_latencies.append(r_latency)
            retrieval_scores.append(r_score)
            if hit:
                retrieval_hits += 1

        ex_with_ctx = MCQExample(
            id=ex.id,
            question=ex.question,
            options=ex.options,
            answer=ex.answer,
            dataset=ex.dataset,
            context=context_text or ex.context,
        )

        prompt = format_prompt(ex_with_ctx)

        t0 = time.perf_counter()
        try:
            output = provider.generate(prompt)
        except Exception as exc:
            output = f"ERROR: {exc}"
        latency = time.perf_counter() - t0

        parsed = parse_answer(output, ex.options)
        correct: bool | None = (parsed == ex.answer) if parsed is not None else None

        ds = per_dataset[ex.dataset]
        ds["total"] += 1
        if parsed is None:
            ds["invalid"] += 1
        elif correct:
            ds["correct"] += 1

        results.append(
            BenchmarkResult(
                example_id=ex.id,
                dataset=ex.dataset,
                question=ex.question,
                options=ex.options,
                gold_answer=ex.answer,
                model_output=output,
                parsed_answer=parsed,
                correct=correct,
                latency_s=latency,
                prompt=prompt,
                retrieval_hit=hit,
                retrieval_score=r_score,
                retrieval_latency_s=r_latency,
                kg_path_count=kg_paths_count,
                kg_latency_s=kg_latency,
                linked_entities=linked_entity_names,
            )
        )

        if verbose:
            status = "✓" if correct else ("?" if correct is None else "✗")
            r_tag = ""
            if active_retriever is not None:
                r_tag = f" [ret={r_score:.2f}"
                if kg_active:
                    r_tag += f" kg={kg_paths_count}"
                r_tag += "]"
            print(f"  [{idx + 1}/{len(examples)}] {ex.dataset} {status} ({latency:.2f}s){r_tag}")

    total = len(results)
    n_correct = sum(1 for r in results if r.correct)
    n_invalid = sum(1 for r in results if r.parsed_answer is None)
    n_valid = total - n_invalid
    latencies = [r.latency_s for r in results]

    for ds_stats in per_dataset.values():
        n_ds_valid = ds_stats["total"] - ds_stats["invalid"]
        ds_stats["accuracy"] = ds_stats["correct"] / n_ds_valid if n_ds_valid > 0 else 0.0

    r_diagnostics: RetrievalDiagnostics | None = None
    if active_retriever is not None:
        r_diagnostics = RetrievalDiagnostics(
            retriever=retriever_name,
            corpus_hit_rate=retrieval_hits / total if total else 0.0,
            avg_score=sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0,
            avg_latency_s=(
                sum(retrieval_latencies) / len(retrieval_latencies)
                if retrieval_latencies
                else 0.0
            ),
            kg_hit_rate=(kg_hits / total) if (kg_active and total) else None,
            avg_kg_paths=(
                sum(kg_path_counts) / len(kg_path_counts)
                if (kg_active and kg_path_counts)
                else None
            ),
            avg_kg_latency_s=(
                sum(kg_latencies) / len(kg_latencies)
                if (kg_active and kg_latencies)
                else None
            ),
            n_conflicts=len(all_conflicts) if (kg_active and conflict_reporter is not None) else None,
        )

    report = BenchmarkReport(
        suite=suite,
        model=provider.name,
        config=config,
        config_hash=c_hash,
        timestamp=timestamp,
        total=total,
        correct=n_correct,
        invalid=n_invalid,
        accuracy=n_correct / n_valid if n_valid > 0 else 0.0,
        invalid_rate=n_invalid / total if total > 0 else 0.0,
        mean_latency_s=sum(latencies) / len(latencies) if latencies else 0.0,
        per_dataset=dict(per_dataset),
        results=results,
        retrieval_diagnostics=r_diagnostics,
    )

    jsonl_path, md_path = write_results(report, Path(output_dir))

    if all_conflicts and conflict_output_dir is not None:
        from minimed_rag.kg_build.runtime_conflict_reporter import write_conflicts_jsonl

        conflict_path = write_conflicts_jsonl(
            all_conflicts, conflict_output_dir, suite, timestamp
        )
        if verbose:
            print(f"Conflicts: {len(all_conflicts)} → {conflict_path}")

    if verbose:
        print(f"\nResults written to:\n  {jsonl_path}\n  {md_path}")

    return report
