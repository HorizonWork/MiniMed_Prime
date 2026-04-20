from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.layers.layer2_embedder import MedicalGraphEmbedder
from src.schemas import EvidenceBundle
from src.training import TRMTrainingExample, build_trm_dataset_arrays, parse_reasoning_chain, save_trm_dataset
from src.utils.kaggle_env import KaggleEnv, T4Hardening
from src.utils.structured_logger import DEFAULT_LOG_DIR, StructuredLogger


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a medical TRM dataset from EvidenceBundle JSONL records.")
    parser.add_argument("--input-jsonl", required=True, type=Path, help="JSONL with evidence plus gold edges/reasoning.")
    parser.add_argument("--output-dir", default=Path("data/trm_medical"), type=Path, help="Output dataset root.")
    parser.add_argument("--split", default="train", choices=("train", "test", "val"), help="Dataset split folder name.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max examples.")
    parser.add_argument("--device", default="cpu", help="Embedder device.")
    parser.add_argument("--primekg-path", default=Path("data/kg/primekg"), type=Path, help="PrimeKG source path for codebook compatibility checks.")
    parser.add_argument("--frozen-codebook", default=None, type=Path, help="Path to a frozen VQ codebook artifact directory or codebook.pt.")
    parser.add_argument(
        "--allow-codebook-fallback",
        action="store_true",
        help="Explicitly allow dataset building without a frozen codebook artifact.",
    )
    parser.add_argument("--embedder-hidden-dim", type=int, default=256, help="Layer 2 hidden dimension.")
    parser.add_argument("--embedder-codebook-size", type=int, default=4096, help="Layer 2 codebook size.")
    parser.add_argument("--embedder-max-len", type=int, default=256, help="Layer 2 max token length.")
    parser.add_argument("--kaggle", action="store_true", help="Enable Kaggle memory/path handling.")
    args = parser.parse_args()

    if args.kaggle or KaggleEnv.is_kaggle():
        T4Hardening.setup_memory()
    if args.frozen_codebook is None and not args.allow_codebook_fallback:
        raise SystemExit(
            "Dataset build now requires a frozen VQ codebook. Pass --frozen-codebook PATH or opt into "
            "--allow-codebook-fallback explicitly."
        )

    event_logger = StructuredLogger("build_trm_dataset", DEFAULT_LOG_DIR)
    examples = list(_load_examples(args.input_jsonl, limit=args.limit))
    embedder = MedicalGraphEmbedder(
        hidden_dim=args.embedder_hidden_dim,
        codebook_size=args.embedder_codebook_size,
        max_len=args.embedder_max_len,
        sapbert_model_name=_optional_path("data/checkpoints/sapbert"),
        medcpt_article_model_name=_optional_path("data/checkpoints/medcpt-article"),
        device=args.device,
        primekg_path=args.primekg_path,
        frozen_codebook_path=args.frozen_codebook,
        allow_codebook_fallback=args.allow_codebook_fallback,
    )
    arrays = build_trm_dataset_arrays(examples, embedder)
    save_dir = save_trm_dataset(arrays, KaggleEnv.path(args.output_dir), split=args.split)
    label_stats = dict(arrays.metadata.get("label_stats", {}))
    graph_stats = dict(arrays.metadata.get("graph_stats", {}))
    report = {
        "status": "SUCCESS" if arrays.inputs.shape[0] > 0 else "EMPTY",
        "examples_saved": int(arrays.inputs.shape[0]),
        "examples_skipped": len(arrays.skipped_examples),
        "valid_rows": int(label_stats.get("rows_valid", arrays.inputs.shape[0])),
        "rejected_rows": int(label_stats.get("rows_rejected", len(arrays.skipped_examples))),
        "invalid_label_reject_count": int(label_stats.get("invalid_label_reject_count", 0)),
        "invalid_label_reject_rate": float(label_stats.get("invalid_label_reject_rate", 0.0)),
        "mean_labeled_token_count": float(graph_stats.get("mean_labeled_token_count", 0.0)),
        "label_mode": label_stats.get("label_mode"),
        "save_dir": str(save_dir),
        "codebook_version": getattr(embedder, "codebook_version", None),
        "frozen_codebook_path": getattr(embedder, "frozen_codebook_path", None),
        "metadata": arrays.metadata,
    }
    event_logger.log_event(
        "dataset_saved",
        {
            **report,
            "backend_used": "medical_trm_dataset_builder",
            "latency_ms": 0.0,
        },
    )
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if arrays.inputs.shape[0] > 0 else 1


def _load_examples(path: Path, limit: int | None = None) -> list[TRMTrainingExample]:
    examples: list[TRMTrainingExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if limit is not None and len(examples) >= limit:
                break
            if not line.strip():
                continue
            payload = json.loads(line)
            evidence_payload = payload.get("evidence") or payload.get("evidence_bundle") or payload
            evidence = EvidenceBundle.model_validate(evidence_payload)
            gold_edge_ids = payload.get("gold_edge_ids") or payload.get("edge_ids") or parse_reasoning_chain(payload.get("reasoning") or payload.get("cot") or payload)
            examples.append(
                TRMTrainingExample(
                    evidence=evidence,
                    gold_edge_ids=[str(edge_id).removeprefix("edge:").removeprefix("edge_id:") for edge_id in gold_edge_ids],
                    group_id=str(payload.get("group_id") or payload.get("question_id") or line_index),
                    metadata={"line_index": line_index},
                )
            )
    return examples


def _optional_path(path: str) -> str | None:
    resolved = KaggleEnv.path(path)
    return str(resolved) if resolved.exists() else None


if __name__ == "__main__":
    raise SystemExit(main())
