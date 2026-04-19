from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.layers.layer3_trm import TRMReasoner
from src.utils.kaggle_env import KaggleEnv, T4Hardening
from src.utils.structured_logger import DEFAULT_LOG_DIR, StructuredLogger


DEFAULT_TRM_PATH = "external/TinyRecursiveModels"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Layer 3 TRM readiness without running the full pipeline.")
    parser.add_argument("--model-path", default=DEFAULT_TRM_PATH, help="TRM repo or checkpoint path.")
    parser.add_argument("--device", default="cpu", help="Device for the readiness check.")
    parser.add_argument("--kaggle", action="store_true", help="Enable Kaggle T4 memory hardening.")
    parser.add_argument(
        "--trm-nonstrict",
        action="store_true",
        help="Allow TRM load to continue even if dummy forward validation fails.",
    )
    parser.add_argument(
        "--allow-official-untrained",
        action="store_true",
        help="Initialize the official Samsung TRM core from scratch for fine-tuning readiness checks.",
    )
    args = parser.parse_args()

    if args.kaggle or KaggleEnv.is_kaggle():
        T4Hardening.setup_memory()

    check_logger = StructuredLogger("check_trm_ready", DEFAULT_LOG_DIR)
    try:
        reasoner = TRMReasoner(
            model_path=args.model_path,
            device=args.device,
            max_steps=1,
            allow_official_untrained=args.allow_official_untrained,
            strict_validate=not args.trm_nonstrict,
        )
        bundle = reasoner.model_bundle
        report: dict[str, Any] = {
            "status": "READY" if reasoner.trm_is_real else "NEEDS_MEDICAL_FINETUNE",
            "trm_is_real": reasoner.trm_is_real,
            "backend_used": reasoner.backend_used,
            "source": bundle.source,
            "checkpoint_path": str(bundle.checkpoint_path) if bundle.checkpoint_path is not None else None,
            "checkpoint_loaded": bundle.checkpoint_loaded,
            "medical_compatible": bundle.medical_compatible,
            "repo_root": str(bundle.repo_root) if bundle.repo_root is not None else None,
            "config": {
                "seq_len": bundle.config.seq_len,
                "vocab_size": bundle.config.vocab_size,
                "num_puzzle_identifiers": bundle.config.num_puzzle_identifiers,
                "hidden_size": bundle.config.hidden_size,
                "num_heads": bundle.config.num_heads,
                "L_layers": bundle.config.L_layers,
                "H_cycles": bundle.config.H_cycles,
                "L_cycles": bundle.config.L_cycles,
                "forward_dtype": bundle.config.forward_dtype,
            },
            "load_diagnostics": list(bundle.load_diagnostics),
        }
        check_logger.log_event(
            "trm_ready_check",
            {
                "backend_used": reasoner.backend_used,
                "trm_is_real": reasoner.trm_is_real,
                "checkpoint_loaded": bundle.checkpoint_loaded,
                "medical_compatible": bundle.medical_compatible,
                "latency_ms": 0.0,
            },
        )
    except Exception as exc:
        check_logger.log_exception(
            "trm_ready_check_failed",
            exc,
            {"backend_used": "trm_check", "latency_ms": 0.0},
        )
        report = {
            "status": "FAILED",
            "trm_is_real": False,
            "backend_used": "unavailable",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["status"] in {"READY", "NEEDS_MEDICAL_FINETUNE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
