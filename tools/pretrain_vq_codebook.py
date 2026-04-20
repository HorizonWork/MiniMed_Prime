from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.layers.layer2_embedder import MedicalGraphEmbedder
from src.layers.vq_artifacts import (
    append_train_log,
    estimate_token_stability,
    inspect_codebook_usage,
    resolve_artifact_paths,
    save_codebook_artifact,
    utc_now_iso,
)
from src.layers.vq_dataset import PrimeKGVQDataset
from src.utils.kaggle_env import KaggleEnv, T4Hardening

DEFAULT_LR = 1e-3
DEFAULT_PRIMEKG_PATH = Path("data/kg/primekg")
DEFAULT_SAPBERT_PATH = Path("data/checkpoints/sapbert")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretrain and version a frozen VQ codebook for MiniMed Prime.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory where VQ artifacts will be written.")
    parser.add_argument("--num-subgraphs", type=int, default=50000, help="Number of PrimeKG random subgraphs to sample.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of passes over the deterministic sample stream.")
    parser.add_argument("--batch-size", type=int, default=128, help="Number of subgraphs per optimizer step.")
    parser.add_argument("--codebook-size", type=int, default=4096, help="Number of VQ tokens including the padding id.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension used by the graph embedder and VQ.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing artifact in output-dir.")
    parser.add_argument("--save-every", type=int, default=1000, help="Persist progress every N subgraphs.")
    parser.add_argument("--max-runtime-min", type=float, default=None, help="Optional wall-clock budget in minutes.")
    parser.add_argument("--kaggle", action="store_true", help="Enable Kaggle runtime hardening.")
    parser.add_argument("--dry-run", type=int, default=None, help="Override num-subgraphs with a smaller smoke-test value.")
    parser.add_argument("--primekg-path", type=Path, default=DEFAULT_PRIMEKG_PATH, help="PrimeKG path used for deterministic random subgraph sampling.")
    parser.add_argument("--sapbert-path", type=Path, default=DEFAULT_SAPBERT_PATH, help="Optional SapBERT path. If missing, hash fallback encoder is used.")
    parser.add_argument(
        "--max-primekg-edges",
        type=int,
        default=None,
        help="Optional cap on loaded PrimeKG edges. Useful for dry-run in low-memory environments.",
    )
    args = parser.parse_args()

    if args.kaggle or KaggleEnv.is_kaggle():
        T4Hardening.setup_memory()

    effective_num_subgraphs = int(args.dry_run if args.dry_run is not None else args.num_subgraphs)
    if effective_num_subgraphs <= 0:
        raise SystemExit("--num-subgraphs/--dry-run must be positive.")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.save_every <= 0:
        raise SystemExit("--save-every must be positive.")

    output_dir = KaggleEnv.ensure_writeable(args.output_dir)
    paths = resolve_artifact_paths(output_dir)
    max_primekg_edges = args.max_primekg_edges
    if max_primekg_edges is None and args.dry_run is not None:
        max_primekg_edges = 50_000
    dataset = PrimeKGVQDataset(
        primekg_path=args.primekg_path,
        seed=args.seed,
        max_edges_to_load=max_primekg_edges,
    )
    embedder = MedicalGraphEmbedder(
        hidden_dim=args.hidden_dim,
        codebook_size=args.codebook_size,
        max_len=256,
        sapbert_model_name=_optional_path(args.sapbert_path),
        medcpt_article_model_name=None,
        device="cpu",
        primekg_path=dataset.primekg_path,
        allow_codebook_fallback=True,
    )

    total_target = int(effective_num_subgraphs * args.epochs)
    resume_state = _load_resume_state(paths) if args.resume else None
    created_at = resume_state["created_at"] if resume_state is not None else utc_now_iso()
    codebook_version = resume_state["codebook_version"] if resume_state is not None else embedder.build_codebook_metadata(
        num_subgraphs_seen=0,
        seed=args.seed,
        created_at=created_at,
    )["codebook_version"]
    subgraphs_seen = int(resume_state["subgraphs_seen"]) if resume_state is not None else 0
    elapsed_before = float(resume_state["elapsed_sec"]) if resume_state is not None else 0.0
    usage_counts_total = (
        torch.tensor(resume_state["usage_counts_total"], dtype=torch.long)
        if resume_state is not None
        else torch.zeros((args.codebook_size - 1,), dtype=torch.long)
    )
    vectors_seen = int(resume_state["vectors_seen"]) if resume_state is not None else 0

    if resume_state is not None:
        _validate_resume_args(resume_state, args)
        embedder.load_frozen_codebook(paths.output_dir, freeze=False, strict_source_graph_signature=False)

    optimizer = torch.optim.Adam([embedder.vq.codebook.weight, *embedder.reconstruction_head.parameters()], lr=DEFAULT_LR)
    started_at = time.perf_counter()
    last_snapshot: dict[str, Any] = {
        "status": "initialized",
        "subgraphs_seen": subgraphs_seen,
        "vectors_seen": vectors_seen,
        "loss": None,
        "reconstruction_loss": None,
        "commitment_loss": None,
        "dead_code_ratio": float((usage_counts_total == 0).sum().item() / max(usage_counts_total.numel(), 1)),
        "updated_at": utc_now_iso(),
        "elapsed_sec": elapsed_before,
    }

    _persist_snapshot(
        output_dir=paths.output_dir,
        embedder=embedder,
        args=args,
        created_at=created_at,
        codebook_version=codebook_version,
        subgraphs_seen=subgraphs_seen,
        total_target=total_target,
        vectors_seen=vectors_seen,
        usage_counts_total=usage_counts_total,
        snapshot=last_snapshot,
        runtime_elapsed=elapsed_before,
        status="running" if subgraphs_seen < total_target else "completed",
    )
    last_saved_subgraphs = subgraphs_seen

    while subgraphs_seen < total_target:
        if _runtime_limit_reached(started_at=started_at, elapsed_before=elapsed_before, max_runtime_min=args.max_runtime_min):
            last_snapshot["status"] = "stopped_max_runtime"
            break

        batch_end = min(subgraphs_seen + args.batch_size, total_target)
        batch_embeddings: list[torch.Tensor] = []
        batch_indices: list[int] = []
        for sample_index in range(subgraphs_seen, batch_end):
            evidence = dataset.sample(sample_index).evidence
            embeddings = embedder._extract_pretraining_embeddings(evidence)
            if embeddings is None or embeddings.numel() == 0:
                continue
            batch_embeddings.append(embeddings.to(embedder.device))
            batch_indices.append(sample_index)

        if not batch_embeddings:
            subgraphs_seen = batch_end
            last_snapshot = {
                "status": "skipped_empty_batch",
                "subgraphs_seen": subgraphs_seen,
                "vectors_seen": vectors_seen,
                "loss": 0.0,
                "reconstruction_loss": 0.0,
                "commitment_loss": 0.0,
                "dead_code_ratio": float((usage_counts_total == 0).sum().item() / max(usage_counts_total.numel(), 1)),
                "updated_at": utc_now_iso(),
                "elapsed_sec": elapsed_before + (time.perf_counter() - started_at),
                "batch_sample_indices": batch_indices,
            }
            append_train_log(paths.output_dir, last_snapshot)
            continue

        batch_vectors = torch.cat(batch_embeddings, dim=0)
        optimizer.zero_grad(set_to_none=True)
        quantized, code_ids, commitment_loss = embedder.vq(batch_vectors)
        reconstruction = embedder.reconstruction_head(quantized)
        reconstruction_loss = F.mse_loss(reconstruction, batch_vectors)
        loss = reconstruction_loss + commitment_loss
        loss.backward()
        optimizer.step()

        usage_counts_total += torch.bincount(code_ids.detach().cpu(), minlength=usage_counts_total.numel())
        subgraphs_seen = batch_end
        vectors_seen += int(batch_vectors.size(0))
        usage_snapshot = inspect_codebook_usage(batch_vectors.detach().cpu(), embedder.vq.codebook.weight.detach().cpu(), top_k=10)
        stability_snapshot = estimate_token_stability(
            batch_vectors.detach().cpu()[: min(512, batch_vectors.size(0))],
            embedder.vq.codebook.weight.detach().cpu(),
            repeats=2,
        )
        runtime_elapsed = elapsed_before + (time.perf_counter() - started_at)
        last_snapshot = {
            "status": "running",
            "subgraphs_seen": subgraphs_seen,
            "vectors_seen": vectors_seen,
            "loss": float(loss.detach().cpu()),
            "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
            "commitment_loss": float(commitment_loss.detach().cpu()),
            "unique_codes_used": int((usage_counts_total > 0).sum().item()),
            "dead_code_ratio": float((usage_counts_total == 0).sum().item() / max(usage_counts_total.numel(), 1)),
            "updated_at": utc_now_iso(),
            "elapsed_sec": runtime_elapsed,
            "batch_sample_indices": batch_indices,
            "batch_usage": usage_snapshot,
            "token_stability": stability_snapshot,
        }
        append_train_log(paths.output_dir, last_snapshot)
        if subgraphs_seen >= last_saved_subgraphs + args.save_every or subgraphs_seen >= total_target:
            _persist_snapshot(
                output_dir=paths.output_dir,
                embedder=embedder,
                args=args,
                created_at=created_at,
                codebook_version=codebook_version,
                subgraphs_seen=subgraphs_seen,
                total_target=total_target,
                vectors_seen=vectors_seen,
                usage_counts_total=usage_counts_total,
                snapshot=last_snapshot,
                runtime_elapsed=runtime_elapsed,
                status="running" if subgraphs_seen < total_target else "completed",
            )
            last_saved_subgraphs = subgraphs_seen

    final_runtime = elapsed_before + (time.perf_counter() - started_at)
    final_status = "completed" if subgraphs_seen >= total_target else str(last_snapshot.get("status") or "stopped")
    _persist_snapshot(
        output_dir=paths.output_dir,
        embedder=embedder,
        args=args,
        created_at=created_at,
        codebook_version=codebook_version,
        subgraphs_seen=subgraphs_seen,
        total_target=total_target,
        vectors_seen=vectors_seen,
        usage_counts_total=usage_counts_total,
        snapshot=last_snapshot,
        runtime_elapsed=final_runtime,
        status=final_status,
    )
    report = {
        "status": final_status,
        "output_dir": str(paths.output_dir),
        "codebook_version": codebook_version,
        "subgraphs_seen": subgraphs_seen,
        "target_subgraphs": total_target,
        "vectors_seen": vectors_seen,
        "max_primekg_edges": max_primekg_edges,
        "dead_code_ratio": float((usage_counts_total == 0).sum().item() / max(usage_counts_total.numel(), 1)),
        "elapsed_sec": final_runtime,
    }
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if final_status == "completed" else 1


def _persist_snapshot(
    *,
    output_dir: Path,
    embedder: MedicalGraphEmbedder,
    args: argparse.Namespace,
    created_at: str,
    codebook_version: str,
    subgraphs_seen: int,
    total_target: int,
    vectors_seen: int,
    usage_counts_total: torch.Tensor,
    snapshot: dict[str, Any],
    runtime_elapsed: float,
    status: str,
) -> None:
    top_count = min(10, usage_counts_total.numel())
    top_values, top_indices = torch.topk(usage_counts_total, k=top_count) if top_count else (
        torch.zeros((0,), dtype=torch.long),
        torch.zeros((0,), dtype=torch.long),
    )
    top_used_clusters = [
        {
            "cluster_id": int(index.item()),
            "count": int(value.item()),
            "share": float(value.item() / max(vectors_seen, 1)),
        }
        for value, index in zip(top_values, top_indices)
        if int(value.item()) > 0
    ]
    unused_clusters = torch.nonzero(usage_counts_total == 0, as_tuple=False).flatten().tolist()
    metadata = embedder.build_codebook_metadata(
        num_subgraphs_seen=subgraphs_seen,
        seed=args.seed,
        created_at=created_at,
        codebook_version=codebook_version,
    )
    progress = {
        "status": status,
        "num_subgraphs_target": int(total_target),
        "subgraphs_seen": int(subgraphs_seen),
        "vectors_seen": int(vectors_seen),
        "max_primekg_edges": int(args.max_primekg_edges) if args.max_primekg_edges is not None else None,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "save_every": int(args.save_every),
        "elapsed_sec": float(runtime_elapsed),
        "updated_at": utc_now_iso(),
        "loss": snapshot.get("loss"),
        "reconstruction_loss": snapshot.get("reconstruction_loss"),
        "commitment_loss": snapshot.get("commitment_loss"),
        "dead_code_ratio": snapshot.get("dead_code_ratio"),
    }
    stats = {
        "codebook_version": codebook_version,
        "updated_at": utc_now_iso(),
        "elapsed_sec": float(runtime_elapsed),
        "subgraphs_seen": int(subgraphs_seen),
        "vectors_seen": int(vectors_seen),
        "unique_codes_used": int((usage_counts_total > 0).sum().item()),
        "dead_code_ratio": float((usage_counts_total == 0).sum().item() / max(usage_counts_total.numel(), 1)),
        "top_used_clusters": top_used_clusters,
        "unused_clusters": [int(index) for index in unused_clusters[:50]],
        "usage_counts_total": usage_counts_total.tolist(),
        "last_batch_usage": snapshot.get("batch_usage"),
        "last_token_stability": snapshot.get("token_stability"),
        "last_loss": snapshot.get("loss"),
        "last_reconstruction_loss": snapshot.get("reconstruction_loss"),
        "last_commitment_loss": snapshot.get("commitment_loss"),
    }
    save_codebook_artifact(
        output_dir,
        codebook_weight=embedder.vq.codebook.weight.detach().cpu(),
        metadata=metadata,
        stats=stats,
        progress=progress,
        extra_state=embedder.export_codebook_state(),
    )


def _load_resume_state(paths: Any) -> dict[str, Any] | None:
    if not paths.codebook_path.exists():
        raise SystemExit(f"--resume was set but no existing artifact was found under {paths.output_dir}.")
    metadata = json.loads(paths.meta_path.read_text(encoding="utf-8"))
    progress = json.loads(paths.progress_path.read_text(encoding="utf-8")) if paths.progress_path.exists() else {}
    stats = json.loads(paths.stats_path.read_text(encoding="utf-8")) if paths.stats_path.exists() else {}
    return {
        "created_at": metadata.get("created_at") or utc_now_iso(),
        "codebook_version": metadata.get("codebook_version"),
        "seed": metadata.get("seed"),
        "codebook_size": metadata.get("codebook_size"),
        "hidden_dim": metadata.get("hidden_dim"),
        "subgraphs_seen": progress.get("subgraphs_seen", metadata.get("num_subgraphs_seen", 0)),
        "elapsed_sec": progress.get("elapsed_sec", 0.0),
        "usage_counts_total": stats.get("usage_counts_total", [0] * (int(metadata.get("codebook_size", 1)) - 1)),
        "vectors_seen": stats.get("vectors_seen", 0),
    }


def _validate_resume_args(resume_state: dict[str, Any], args: argparse.Namespace) -> None:
    if int(resume_state.get("seed", args.seed)) != int(args.seed):
        raise SystemExit(
            f"Resume seed mismatch: artifact seed={resume_state.get('seed')} but --seed={args.seed}. "
            "Use the original seed to preserve deterministic sample order."
        )
    if int(resume_state.get("codebook_size", args.codebook_size)) != int(args.codebook_size):
        raise SystemExit(
            "Resume codebook-size mismatch: "
            f"artifact={resume_state.get('codebook_size')} --codebook-size={args.codebook_size}."
        )
    if int(resume_state.get("hidden_dim", args.hidden_dim)) != int(args.hidden_dim):
        raise SystemExit(
            f"Resume hidden-dim mismatch: artifact={resume_state.get('hidden_dim')} --hidden-dim={args.hidden_dim}."
        )


def _runtime_limit_reached(*, started_at: float, elapsed_before: float, max_runtime_min: float | None) -> bool:
    if max_runtime_min is None:
        return False
    return elapsed_before + (time.perf_counter() - started_at) >= float(max_runtime_min) * 60.0


def _optional_path(path: str | Path) -> str | None:
    resolved = KaggleEnv.path(path)
    return str(resolved) if resolved.exists() else None


if __name__ == "__main__":
    raise SystemExit(main())
