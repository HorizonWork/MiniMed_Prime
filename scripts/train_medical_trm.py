#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import TRMTrainerConfig, train_medical_trm
from src.utils.kaggle_env import KaggleEnv, T4Hardening


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/fine-tune the medical TRM on prepared GraphTensor arrays.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/trm_medical"), help="Root containing train/val TRM .npy splits.")
    parser.add_argument("--train-split", default="train", help="Train split directory name under dataset-dir.")
    parser.add_argument("--val-split", default="val", help="Validation split directory name; use 'none' to disable.")
    parser.add_argument("--model-path", default="external/TinyRecursiveModels", help="Samsung TRM repo/checkpoint path.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/checkpoints/active"), help="Writable checkpoint output dir.")
    parser.add_argument("--final-checkpoint-name", default="medical_trm.pt", help="Final medical-compatible checkpoint filename.")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or cuda device string.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=8, help="TRM recursive supervision steps during training.")
    parser.add_argument("--save-every", type=int, default=2500)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-official-untrained", action="store_true", help="Do not initialize Samsung TRM from scratch if no checkpoint loads.")
    parser.add_argument("--kaggle", action="store_true", help="Enable Kaggle/T4 memory hardening.")
    args = parser.parse_args()

    if args.kaggle or KaggleEnv.is_kaggle():
        T4Hardening.setup_memory()
        T4Hardening.verify_fp16("float16")

    device = _resolve_device(args.device)
    val_split = None if args.val_split.lower() in {"", "none", "null", "false"} else args.val_split
    config = TRMTrainerConfig(
        dataset_dir=args.dataset_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=val_split,
        final_checkpoint_name=args.final_checkpoint_name,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        save_every=args.save_every,
        eval_every=args.eval_every,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        allow_official_untrained=not args.no_official_untrained,
        seed=args.seed,
    )
    result = train_medical_trm(config)
    sys.stdout.write(json.dumps(result.to_json_dict(), indent=2, sort_keys=True) + "\n")
    return 0 if result.status == "SUCCESS" else 1


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


if __name__ == "__main__":
    raise SystemExit(main())
