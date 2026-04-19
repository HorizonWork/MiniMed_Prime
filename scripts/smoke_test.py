from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.layers.layer3_trm import TRMReasoner
from src.orchestrator import MedicalReasoningSystemV3, SystemConfig
from src.utils.kaggle_env import KaggleEnv, T4Hardening
from src.utils.path_resolver import KagglePathResolver
from src.utils.structured_logger import DEFAULT_LOG_DIR, StructuredLogger

try:
    import torch
except ImportError:  # pragma: no cover - torch is part of the project stack
    torch = None

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())


SMOKE_TEST_QUESTION = (
    "A 68-year-old male with atrial fibrillation and newly diagnosed peptic ulcer disease is currently on warfarin. "
    "Which of the following antibiotics for H. pylori eradication poses the highest bleeding risk? "
    "(A) Amoxicillin (B) Clarithromycin (C) Metronidazole (D) Doxycycline"
)
SMOKE_OUTPUT_PATH = KaggleEnv.ensure_writeable(KaggleEnv.path("data/smoke_test_output.json"))
PREFERRED_TRM_CHECKPOINTS = (
    "medical_trm.pt",
    "medical_trm.ckpt",
    "medical_trm_finetuned.pt",
    "medical_trm_finetuned.ckpt",
    "minimed_trm_medical.pt",
    "minimed_trm_medical.ckpt",
    "trm_arc_v1_public_step_518071.pt",
    "sudoku_extreme.pt",
    "sudoku_extreme.ckpt",
    "sudoku-extreme.pt",
    "sudoku-extreme.ckpt",
)
DEFAULT_JUDGE_MODEL_ENV = "MINIMED_JUDGE_MODEL"
DEFAULT_SYNTHESIS_MODEL_ENV = "MINIMED_SYNTHESIS_MODEL"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single end-to-end smoke test for MiniMed Prime.")
    parser.add_argument("--kaggle", action="store_true", help="Enable Kaggle-oriented path resolution and output handling.")
    parser.add_argument(
        "--trm-nonstrict",
        action="store_true",
        help="Allow TRM load to continue even when dummy forward validation fails.",
    )
    args = parser.parse_args()

    smoke_logger = StructuredLogger("smoke_test", DEFAULT_LOG_DIR)
    if args.kaggle or KaggleEnv.is_kaggle():
        T4Hardening.setup_memory()

    config = build_system_config(trm_strict_validate=not args.trm_nonstrict)
    preflight = pre_flight_check(config)
    if not preflight.get("all_ready", False):
        logger.warning("Pre-flight check failed, some components may use fallback. Details: %s", preflight)
    system = MedicalReasoningSystemV3(config)
    latency_ms: dict[str, float] = {}
    _wrap_pipeline_step(system, "_retrieve_with_fallback", "layer1", latency_ms)
    _wrap_pipeline_step(system, "_embed", "layer2", latency_ms)
    _wrap_pipeline_step(system, "_reason_with_retry", "layer3", latency_ms)
    _wrap_pipeline_step(system, "_run_preflight_judges", "layer4", latency_ms)
    _wrap_pipeline_step(system, "_synthesize_with_retry", "layer5", latency_ms)

    pipeline_start = time.perf_counter()
    answer = system.answer(SMOKE_TEST_QUESTION)
    latency_ms["total"] = (time.perf_counter() - pipeline_start) * 1000.0
    for layer_name in ("layer1", "layer2", "layer3", "layer4", "layer5"):
        latency_ms.setdefault(layer_name, 0.0)

    SMOKE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SMOKE_OUTPUT_PATH.write_text(answer.model_dump_json(indent=2), encoding="utf-8")

    state = system.last_state
    evidence = state.evidence if state is not None else None
    trm_output = state.trm_output if state is not None else None
    layer1_backend = evidence.metadata.get("entity_linker_backend") if evidence is not None else None
    layer2_backend = dict(getattr(system.embedder, "last_backend_used", {}))
    layer3_backend = getattr(system.trm, "backend_used", None)
    layer4_backends = _judge_backends(system)
    layer5_backend = getattr(system.synthesizer, "last_backend_used", getattr(system.synthesizer, "backend", None))

    components_real = {
        "layer1": layer1_backend == "scispacy_umls",
        "layer2": layer2_backend.get("layer2_backend") == "sapbert_real",
        "layer3": bool(getattr(system.trm, "trm_is_real", False)),
        "layer4": bool(layer4_backends) and all(backend != "heuristic" for backend in layer4_backends.values()),
        "layer5": layer5_backend in {"llama_cpp", "transformers_4bit", "openai"},
    }
    components_fallback = {layer_name: not is_real for layer_name, is_real in components_real.items()}

    provenance_count = len(answer.provenance)
    status = _status_from_result(
        answer_text=answer.answer_text,
        provenance_count=provenance_count,
        components_real=components_real,
    )
    report = {
        "status": status,
        "components_real": components_real,
        "components_fallback": components_fallback,
        "latency_ms": {key: round(value, 2) for key, value in latency_ms.items()},
        "answer_preview": answer.answer_text[:200],
        "abstention": answer.abstention,
        "provenance_tags_found": provenance_count,
        "trm_paths_found": len(trm_output.ranked_paths) if trm_output is not None else 0,
        "backend_details": {
            "layer1": evidence.metadata.get("backend_used") if evidence is not None else None,
            "layer2": layer2_backend,
            "layer3": {
                "backend_used": layer3_backend,
                "trm_is_real": getattr(system.trm, "trm_is_real", False),
                "checkpoint_path": str(getattr(getattr(system.trm, "model_bundle", None), "checkpoint_path", "") or ""),
                "load_diagnostics": list(getattr(getattr(system.trm, "model_bundle", None), "load_diagnostics", []) or []),
            },
            "layer4": layer4_backends,
            "layer5": {"backend_used": layer5_backend},
        },
        "preflight": preflight,
        "output_path": str(SMOKE_OUTPUT_PATH),
    }
    smoke_logger.log_event(
        "smoke_test_report",
        {
            "status": status,
            "components_real": components_real,
            "components_fallback": components_fallback,
            "latency_ms": latency_ms,
            "abstention": answer.abstention,
            "backend_used": "kaggle" if args.kaggle or KaggleEnv.is_kaggle() else "local",
        },
    )
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


def build_system_config(*, trm_strict_validate: bool = True) -> SystemConfig:
    embedder_device, trm_device, judge_device = _select_devices()
    primekg_path = _resolve_primekg_path()
    sapbert_path = KagglePathResolver.resolve_sapbert()
    sapbert_model_name = sapbert_path
    medcpt_article_model_name = KagglePathResolver.resolve_medcpt() or _resolve_optional_dir(KaggleEnv.path("data/checkpoints/medcpt-article"))
    trm_repo_path = KagglePathResolver.resolve_trm_repo()
    trm_checkpoint_path = KagglePathResolver.resolve_trm_checkpoint()
    trm_model_path = trm_checkpoint_path or trm_repo_path or _resolve_trm_path()
    synthesis_model_path = _resolve_synthesis_model_path()
    judge_model_name = _resolve_judge_model_name()

    return SystemConfig(
        primekg_path=primekg_path,
        pubmed_cache_path=KaggleEnv.ensure_writeable(KaggleEnv.path("data/pubmed_cache.jsonl")),
        pubmed_api_key=os.getenv("NCBI_API_KEY"),
        scispacy_model="en_core_sci_lg",
        sapbert_path=sapbert_path,
        sapbert_model_name=sapbert_model_name,
        medcpt_article_model_name=medcpt_article_model_name,
        embedder_device=embedder_device,
        trm_model_path=trm_model_path,
        trm_repo_path=trm_repo_path,
        trm_checkpoint_path=trm_checkpoint_path,
        trm_device=trm_device,
        trm_strict_validate=trm_strict_validate,
        judge_model_name=judge_model_name,
        judge_device=judge_device,
        synthesis_model_path=synthesis_model_path,
    )


def _wrap_pipeline_step(
    system: MedicalReasoningSystemV3,
    method_name: str,
    metric_key: str,
    latency_ms: dict[str, float],
) -> None:
    original = getattr(system, method_name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            latency_ms[metric_key] = (time.perf_counter() - started_at) * 1000.0

    setattr(system, method_name, wrapped)


def _resolve_primekg_path() -> Path:
    candidate_paths = (
        KaggleEnv.path("data/kg/primekg"),
        KaggleEnv.path("data/primekg.csv"),
        KaggleEnv.path("data"),
    )
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / "data" / "kg" / "primekg"


def _resolve_trm_path() -> str:
    resolved_checkpoint = KagglePathResolver.resolve_trm_checkpoint()
    if resolved_checkpoint:
        return resolved_checkpoint
    resolved_repo_root = KagglePathResolver.resolve_trm_repo()
    if resolved_repo_root:
        return resolved_repo_root

    repo_root = PROJECT_ROOT / "external" / "TinyRecursiveModels"
    if KaggleEnv.is_kaggle():
        candidate_repo = KaggleEnv.path("external/TinyRecursiveModels")
        if candidate_repo.exists():
            repo_root = candidate_repo
    checkpoint_dir = repo_root / "checkpoints"
    if checkpoint_dir.exists():
        for checkpoint_name in PREFERRED_TRM_CHECKPOINTS:
            candidate = checkpoint_dir / checkpoint_name
            if candidate.exists():
                return str(candidate)
        checkpoints = sorted(checkpoint_dir.rglob("*.pt")) + sorted(checkpoint_dir.rglob("*.ckpt"))
        if checkpoints:
            return str(checkpoints[0])
    if repo_root.exists():
        return str(repo_root)
    return "medical_trm.ckpt"


def _resolve_sapbert_path() -> str | None:
    resolved = KagglePathResolver.resolve_sapbert()
    if resolved is not None:
        return resolved
    candidates = (
        KaggleEnv.path("data/checkpoints/sapbert"),
        PROJECT_ROOT / "data" / "checkpoints" / "sapbert",
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def pre_flight_check(config: SystemConfig) -> dict[str, Any]:
    """Verify key real components can initialize before full smoke run."""
    results: dict[str, Any] = {
        "sapbert_path": config.sapbert_path or KagglePathResolver.resolve_sapbert(),
        "trm_checkpoint_path": config.trm_checkpoint_path or KagglePathResolver.resolve_trm_checkpoint(),
        "trm_repo_path": config.trm_repo_path or KagglePathResolver.resolve_trm_repo(),
    }
    results["sapbert_resolved"] = bool(results["sapbert_path"])
    results["trm_checkpoint_resolved"] = bool(results["trm_checkpoint_path"])
    results["trm_repo_resolved"] = bool(results["trm_repo_path"])

    if results["trm_checkpoint_resolved"] or results["trm_repo_resolved"]:
        trm_probe_path = str(results["trm_checkpoint_path"] or results["trm_repo_path"])
        try:
            trm = TRMReasoner(
                model_path=trm_probe_path,
                device=config.trm_device,
                max_steps=1,
                strict_validate=config.trm_strict_validate,
            )
            results["trm_load_test"] = bool(trm.trm_is_real)
            results["trm_backend_used"] = str(trm.backend_used)
            results["trm_load_diagnostics"] = list(getattr(trm.model_bundle, "load_diagnostics", []))
        except Exception as exc:
            results["trm_load_test"] = False
            results["trm_load_error"] = str(exc)
    else:
        results["trm_load_test"] = False
        results["trm_load_error"] = "TRM checkpoint/repo not resolved."

    results["all_ready"] = bool(results["sapbert_resolved"] and results["trm_load_test"])
    logger.info("Pre-flight check: %s", results)
    return results


def _resolve_synthesis_model_path() -> str:
    env_model = os.getenv(DEFAULT_SYNTHESIS_MODEL_ENV)
    if env_model:
        return env_model
    if os.getenv("OPENAI_API_KEY"):
        return DEFAULT_OPENAI_MODEL
    medreason_dir = KaggleEnv.path("data/checkpoints/medreason-8b")
    if medreason_dir.exists():
        return str(medreason_dir)

    llm_dir = KaggleEnv.path("data/checkpoints/llm")
    if llm_dir.exists():
        candidates = sorted(llm_dir.glob("*Q4_K_M*.gguf")) + sorted(llm_dir.glob("*q4_k_m*.gguf"))
        if not candidates:
            candidates = sorted(llm_dir.glob("*.gguf"))
        if candidates:
            return str(candidates[0])
    return "heuristic"


def _resolve_judge_model_name() -> str:
    env_model = os.getenv(DEFAULT_JUDGE_MODEL_ENV)
    if env_model:
        return env_model
    if os.getenv("OPENAI_API_KEY"):
        return DEFAULT_OPENAI_MODEL
    local_dir = KaggleEnv.path("data/checkpoints/judges/medical_o1_verifier_3B")
    if local_dir.exists():
        return str(local_dir)
    return "heuristic"


def _resolve_optional_dir(path: Path) -> str | None:
    return str(path) if path.exists() else None


def _select_devices() -> tuple[str, str, str]:
    if torch is None or not torch.cuda.is_available():
        return "cpu", "cpu", "cpu"
    device_count = torch.cuda.device_count()
    if device_count >= 2:
        return "cuda:1", "cuda:1", "cuda:1"
    return "cuda:0", "cuda:0", "cuda:0"


def _judge_backends(system: MedicalReasoningSystemV3) -> dict[str, str]:
    judge_backends: dict[str, str] = {}
    for judge_name, judge in system.judges.items():
        judge_backends[f"preflight_{judge_name}"] = str(getattr(judge, "backend", "heuristic"))
    hallucination_judge = getattr(system, "hallucination_judge", None)
    if hallucination_judge is not None:
        for attribute_name in (
            "entity_validator",
            "grounding_inspector",
            "soundness_auditor",
            "faithfulness_guardian",
        ):
            judge = getattr(hallucination_judge, attribute_name, None)
            if judge is not None:
                judge_backends[f"wrapper_{attribute_name}"] = str(getattr(judge, "backend", "heuristic"))
    return judge_backends


def _status_from_result(answer_text: str, provenance_count: int, components_real: dict[str, bool]) -> str:
    if not answer_text.strip() or provenance_count < 1:
        return "FAILED"
    if all(components_real.values()):
        return "SUCCESS"
    return "PARTIAL"


if __name__ == "__main__":
    raise SystemExit(main())
