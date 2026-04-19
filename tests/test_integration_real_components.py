from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from src.layers.layer3_trm import TRMReasoner
from src.layers.layer5_synthesis import AnswerSynthesizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRM_REPO_ROOT = PROJECT_ROOT / "external" / "TinyRecursiveModels"
TRM_CHECKPOINT_DIR = TRM_REPO_ROOT / "checkpoints"
MEDREASON_DIR = PROJECT_ROOT / "data" / "checkpoints" / "medreason-8b"
LLM_DIR = PROJECT_ROOT / "data" / "checkpoints" / "llm"
REAL_TRM_CHECKPOINTS = (
    TRM_CHECKPOINT_DIR / "sudoku_extreme.pt",
    TRM_CHECKPOINT_DIR / "sudoku_extreme.ckpt",
    TRM_CHECKPOINT_DIR / "sudoku-extreme.pt",
    TRM_CHECKPOINT_DIR / "sudoku-extreme.ckpt",
)
REAL_GGUF_CANDIDATES = tuple(sorted(LLM_DIR.glob("*Q4_K_M*.gguf"))) + tuple(sorted(LLM_DIR.glob("*q4_k_m*.gguf")))
LLAMA_CPP_AVAILABLE = importlib.util.find_spec("llama_cpp") is not None
TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None
BITSANDBYTES_AVAILABLE = importlib.util.find_spec("bitsandbytes") is not None
ACCELERATE_AVAILABLE = importlib.util.find_spec("accelerate") is not None


def _first_existing_path(candidates: tuple[Path, ...]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


REAL_TRM_CHECKPOINT = _first_existing_path(REAL_TRM_CHECKPOINTS)
REAL_GGUF = REAL_GGUF_CANDIDATES[0] if REAL_GGUF_CANDIDATES else None
REAL_MEDREASON = MEDREASON_DIR if MEDREASON_DIR.exists() else None


@pytest.mark.skipif(
    REAL_TRM_CHECKPOINT is None or not TRM_REPO_ROOT.exists(),
    reason="Samsung TinyRecursiveModels repo/checkpoint not available locally.",
)
def test_real_trm_checkpoint_loads_when_available() -> None:
    reasoner = TRMReasoner(model_path=str(REAL_TRM_CHECKPOINT), device="cpu", max_steps=2)

    assert reasoner.trm_is_real is True
    assert reasoner.backend_used == "samsung_trm"
    assert reasoner.model_bundle.repo_root == TRM_REPO_ROOT


@pytest.mark.skipif(
    REAL_GGUF is None or not LLAMA_CPP_AVAILABLE,
    reason="II-Medical GGUF or llama_cpp is not available locally.",
)
def test_real_synthesis_backend_loads_when_available() -> None:
    synthesizer = AnswerSynthesizer(model_path=str(REAL_GGUF))

    assert synthesizer.backend == "llama_cpp"


@pytest.mark.skipif(
    REAL_MEDREASON is None or not (TRANSFORMERS_AVAILABLE and BITSANDBYTES_AVAILABLE and ACCELERATE_AVAILABLE),
    reason="MedReason-8B or transformers/bitsandbytes/accelerate is not available locally.",
)
def test_transformers_4bit_synthesis_backend_loads_when_available() -> None:
    synthesizer = AnswerSynthesizer(model_path=str(REAL_MEDREASON), backend="transformers_4bit")

    assert synthesizer.backend == "transformers_4bit"
