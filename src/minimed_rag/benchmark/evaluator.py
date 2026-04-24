from __future__ import annotations

import datetime
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .answer_parser import parse_answer
from .datasets.base import MCQExample
from .prompt_templates import format_prompt
from .providers.base import ModelProvider
from .result_writer import BenchmarkReport, BenchmarkResult, RetrievalDiagnostics, write_results

if TYPE_CHECKING:
    from minimed_rag.retrieval.bm25_retriever import RetrievalResult


class ChunkRetriever(Protocol):
    def retrieve(self, query: str, k: int | None = None) -> list["RetrievalResult"]: ...


def _config_hash(config: dict) -> str:
    canon = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


def run_evaluation(
    suite: str,
    examples: list[MCQExample],
    provider: ModelProvider,
    output_dir: str | Path = "artifacts/reports",
    *,
    retriever: ChunkRetriever | None = None,
    verbose: bool = False,
) -> BenchmarkReport:
    retriever_name = type(retriever).__name__ if retriever is not None else "none"
    config = {
        **provider.config,
        "suite": suite,
        "n_examples": len(examples),
        "retriever": retriever_name,
    }
    c_hash = _config_hash(config)
    timestamp = datetime.datetime.utcnow().isoformat()

    results: list[BenchmarkResult] = []
    per_dataset: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0, "invalid": 0})

    # Retrieval diagnostic accumulators
    retrieval_hits = 0
    retrieval_scores: list[float] = []
    retrieval_latencies: list[float] = []

    for idx, ex in enumerate(examples):
        # --- optional retrieval ---
        context_text = ""
        hit = False
        r_score = 0.0
        r_latency = 0.0

        if retriever is not None:
            t_r0 = time.perf_counter()
            retrieved = retriever.retrieve(ex.question)
            r_latency = time.perf_counter() - t_r0

            if retrieved:
                hit = True
                r_score = retrieved[0].score
                from minimed_rag.retrieval.context_builder import ContextBuilder

                context_text = ContextBuilder.build_context_text(retrieved)

            retrieval_latencies.append(r_latency)
            retrieval_scores.append(r_score)
            if hit:
                retrieval_hits += 1

        # Attach retrieved context to the example (MCQExample.context drives prompt)
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
            )
        )

        if verbose:
            status = "✓" if correct else ("?" if correct is None else "✗")
            r_tag = f" [ret={r_score:.2f}]" if retriever else ""
            print(f"  [{idx + 1}/{len(examples)}] {ex.dataset} {status} ({latency:.2f}s){r_tag}")

    total = len(results)
    n_correct = sum(1 for r in results if r.correct)
    n_invalid = sum(1 for r in results if r.parsed_answer is None)
    n_valid = total - n_invalid
    latencies = [r.latency_s for r in results]

    for ds_stats in per_dataset.values():
        n_ds_valid = ds_stats["total"] - ds_stats["invalid"]
        ds_stats["accuracy"] = ds_stats["correct"] / n_ds_valid if n_ds_valid > 0 else 0.0

    # Build retrieval diagnostics block
    r_diagnostics: RetrievalDiagnostics | None = None
    if retriever is not None:
        r_diagnostics = RetrievalDiagnostics(
            retriever=retriever_name,
            corpus_hit_rate=retrieval_hits / total if total else 0.0,
            avg_score=sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.0,
            avg_latency_s=(
                sum(retrieval_latencies) / len(retrieval_latencies)
                if retrieval_latencies
                else 0.0
            ),
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
    if verbose:
        print(f"\nResults written to:\n  {jsonl_path}\n  {md_path}")

    return report
