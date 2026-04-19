from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.layers.layer2_embedder import MedicalGraphEmbedder
from src.layers.layer3_trm import TRMReasoner
from src.schemas import EvidenceBundle
from src.utils.kaggle_env import KaggleEnv, T4Hardening
from src.utils.path_resolver import KagglePathResolver


DEFAULT_TRM_HINT = "/kaggle/working/external/TinyRecursiveModels/checkpoints/trm_arc_v1_public_step_518071.pt"
DEFAULT_SAPBERT_HINT = "/kaggle/input/medv3-checkpoints/sapbert"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify real-vs-fallback status for critical MiniMed components.")
    parser.add_argument("--trm-path", default=DEFAULT_TRM_HINT, help="TRM checkpoint or repo path hint.")
    parser.add_argument("--sapbert-path", default=DEFAULT_SAPBERT_HINT, help="SapBERT path hint.")
    parser.add_argument("--device", default="cpu", help="TRM device: cpu, cuda, or cuda:N")
    parser.add_argument("--kaggle", action="store_true", help="Apply Kaggle/T4 runtime hardening.")
    parser.add_argument(
        "--trm-nonstrict",
        action="store_true",
        help="Allow TRM load to continue even if dummy forward validation fails.",
    )
    args = parser.parse_args()

    if args.kaggle or KaggleEnv.is_kaggle():
        T4Hardening.setup_memory()

    report: dict[str, Any] = {
        "environment": {
            "is_kaggle": KaggleEnv.is_kaggle(),
            "project_root": str(PROJECT_ROOT),
        },
        "paths": {},
        "components": {},
        "suggested_fixes": [],
    }

    trm_checkpoint_resolved = KagglePathResolver.resolve_trm_checkpoint()
    trm_repo_resolved = KagglePathResolver.resolve_trm_repo()
    resolved_sapbert_path = KagglePathResolver.resolve_sapbert()
    trm_checkpoint_path = Path(trm_checkpoint_resolved) if trm_checkpoint_resolved else None
    trm_repo_root = Path(trm_repo_resolved) if trm_repo_resolved else None
    trm_path_source = "resolver" if trm_checkpoint_resolved or trm_repo_resolved else "unresolved"
    if trm_checkpoint_path is None and trm_repo_root is None:
        hinted_path = Path(args.trm_path)
        if hinted_path.exists():
            if hinted_path.is_file():
                trm_checkpoint_path = hinted_path
            else:
                trm_repo_root = hinted_path
            trm_path_source = "arg_hint"
    if resolved_sapbert_path is None:
        explicit = Path(args.sapbert_path)
        if explicit.exists():
            resolved_sapbert_path = str(explicit)
    report["paths"] = {
        "trm_path_source": trm_path_source,
        "trm_repo_root": str(trm_repo_root) if trm_repo_root is not None else None,
        "trm_checkpoint_path": str(trm_checkpoint_path) if trm_checkpoint_path is not None else None,
        "sapbert_path": resolved_sapbert_path,
    }

    report["components"]["layer3_trm"] = _verify_trm(
        trm_checkpoint_path=trm_checkpoint_path,
        trm_repo_root=trm_repo_root,
        fallback_hint=args.trm_path,
        device=args.device,
        strict_validate=not args.trm_nonstrict,
    )
    report["components"]["layer2_embedder"] = _verify_layer2(
        sapbert_path=resolved_sapbert_path,
    )

    _add_suggestions(report)
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


def _verify_trm(
    *,
    trm_checkpoint_path: Path | None,
    trm_repo_root: Path | None,
    fallback_hint: str,
    device: str,
    strict_validate: bool,
) -> dict[str, Any]:
    target_path = str(trm_checkpoint_path or trm_repo_root or fallback_hint)
    component: dict[str, Any] = {
        "component": "layer3_trm",
        "target_path": target_path,
        "trm_is_real": False,
        "backend_used": "uninitialized",
        "status": "fallback",
        "error": None,
        "traceback": None,
        "load_diagnostics": [],
    }
    try:
        reasoner = TRMReasoner(
            model_path=target_path,
            device=device,
            max_steps=2,
            conf_threshold=0.99,
            contradiction_threshold=0.5,
            strict_validate=strict_validate,
        )
        probe_bundle = _build_probe_bundle()
        probe_inputs = torch.zeros((1, 256), dtype=torch.long)
        probe_inputs[0, :2] = torch.tensor([1, 2], dtype=torch.long)
        probe_output = reasoner.reason(
            {
                "inputs": probe_inputs,
                "puzzle_identifiers": torch.tensor([0], dtype=torch.long),
                "node_mapping": {
                    0: "drug:test_drug",
                    1: "disease:test_disease",
                    **{index: "__pad__" for index in range(2, 256)},
                },
                "original_graph": probe_bundle,
            }
        )
        component["trm_is_real"] = bool(reasoner.trm_is_real)
        component["backend_used"] = str(reasoner.backend_used)
        component["status"] = "real" if reasoner.trm_is_real else "fallback"
        component["checkpoint_loaded"] = bool(reasoner.model_bundle.checkpoint_loaded)
        component["checkpoint_path"] = str(reasoner.model_bundle.checkpoint_path) if reasoner.model_bundle.checkpoint_path is not None else None
        component["repo_root"] = str(reasoner.model_bundle.repo_root) if reasoner.model_bundle.repo_root is not None else None
        component["medical_compatible"] = bool(reasoner.model_bundle.medical_compatible)
        component["trace_steps"] = len(probe_output.trace)
        component["load_diagnostics"] = list(reasoner.model_bundle.load_diagnostics)
    except Exception as exc:
        component["error"] = f"{type(exc).__name__}: {exc}"
        component["traceback"] = traceback.format_exc()
    return component


def _verify_layer2(*, sapbert_path: str | None) -> dict[str, Any]:
    component: dict[str, Any] = {
        "component": "layer2_embedder",
        "sapbert_path": sapbert_path,
        "layer2_backend": "fallback_hash",
        "status": "fallback",
        "error": None,
        "traceback": None,
    }
    try:
        bundle = _build_probe_bundle()
        embedder = MedicalGraphEmbedder(
            hidden_dim=64,
            codebook_size=128,
            max_len=256,
            sapbert_path=sapbert_path,
            sapbert_model_name=sapbert_path,
            medcpt_article_model_name=None,
            device="cpu",
        )
        outputs = embedder(bundle)
        layer2_backend = str(getattr(embedder, "backend_used", outputs.get("layer2_backend", "fallback_hash")))
        component["layer2_backend"] = layer2_backend
        component["status"] = "real" if layer2_backend == "sapbert_real" else "fallback"
        component["backend_details"] = dict(outputs.get("backend_used", {}))
    except Exception as exc:
        component["error"] = f"{type(exc).__name__}: {exc}"
        component["traceback"] = traceback.format_exc()
    return component


def _build_probe_bundle() -> EvidenceBundle:
    return EvidenceBundle.model_validate(
        {
            "question_text": "Is test_drug associated with test_disease?",
            "question_type": "factoid",
            "question_entities": [
                {
                    "surface": "test_drug",
                    "cui": "C0000001",
                    "primekg_node_id": "drug:test_drug",
                    "entity_type": "drug",
                },
                {
                    "surface": "test_disease",
                    "cui": "C0000002",
                    "primekg_node_id": "disease:test_disease",
                    "entity_type": "disease",
                },
            ],
            "subgraph_edges": [
                {
                    "edge_id": "E_TEST_1",
                    "head": "drug:test_drug",
                    "tail": "disease:test_disease",
                    "relation": "indication",
                    "display_relation": "is indicated for",
                    "source_reliability": 0.9,
                    "amg_confidence": 0.9,
                    "supporting_pmids": [],
                }
            ],
            "pubmed_passages": [],
            "metadata": {"probe": True},
        }
    )


def _add_suggestions(report: dict[str, Any]) -> None:
    suggestions: list[str] = []
    trm_info = report["components"].get("layer3_trm", {})
    layer2_info = report["components"].get("layer2_embedder", {})

    if trm_info.get("status") != "real":
        suggestions.append(
            "TRM is not real: verify checkpoint path and Samsung repo mount under /kaggle/working/external/TinyRecursiveModels."
        )
        if trm_info.get("error"):
            suggestions.append(f"TRM error: {trm_info['error']}")
    if layer2_info.get("layer2_backend") != "sapbert_real":
        suggestions.append(
            "Layer 2 is fallback: verify SapBERT exists at /kaggle/input/medv3-checkpoints/sapbert and includes config/tokenizer/model files."
        )
        if layer2_info.get("error"):
            suggestions.append(f"Layer 2 error: {layer2_info['error']}")
    if not suggestions:
        suggestions.append("All verified components are using real backends.")
    report["suggested_fixes"] = suggestions


if __name__ == "__main__":
    raise SystemExit(main())
