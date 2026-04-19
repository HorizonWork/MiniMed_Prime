from __future__ import annotations

import importlib
import json
import logging
import math
import sys
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
import yaml
from torch import nn

from src.utils.kaggle_env import KaggleEnv, T4Hardening
from src.utils.path_resolver import KagglePathResolver

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())


DEFAULT_TRM_VOCAB_SIZE = 8192
DEFAULT_TRM_SEQ_LEN = 256
DEFAULT_TRM_PUZZLE_IDENTIFIERS = 5
DEFAULT_TRM_H_CYCLES = 3
DEFAULT_TRM_L_CYCLES = 4
DEFAULT_TRM_LAYERS = 2
DEFAULT_TRM_HIDDEN_SIZE = 512
DEFAULT_TRM_HEADS = 8
DEFAULT_TRM_EXPANSION = 4.0
DEFAULT_TRM_PUZZLE_EMB_LEN = 16
DEFAULT_TRM_FORWARD_DTYPE = "float16"
DEFAULT_TRM_HALT_MAX_STEPS = 16
DEFAULT_IGNORE_LABEL_ID = -100
DEFAULT_EXTERNAL_TRM_REPO = Path("external/TinyRecursiveModels")
DEFAULT_EXTERNAL_TRM_CHECKPOINT_DIR = DEFAULT_EXTERNAL_TRM_REPO / "checkpoints"
CHECKPOINT_CONFIG_NAMES = ("all_config.yaml", "config.yaml")
PREFERRED_TRM_CHECKPOINT_NAMES = (
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
    "sudoku_extreme.pth",
    "sudoku-extreme.pth",
)
OFFICIAL_TRM_CLASS_NAME = "TinyRecursiveReasoningModel_ACTV1"
OFFICIAL_TRM_MODULE = "models.recursive_reasoning.trm"
OFFICIAL_TRM_CONFIG_PATH = Path("config/arch/trm.yaml")
OFFICIAL_TRM_SOURCE_FILE = Path("models/recursive_reasoning/trm.py")
OFFICIAL_TRM_ALT_SOURCE_FILE = Path("src/trm.py")
OFFICIAL_TRM_CLASS_CANDIDATES: tuple[str, ...] = (
    OFFICIAL_TRM_CLASS_NAME,
    "TinyRecursiveModel",
    "TinyRecursiveReasoningModel",
)
KAGGLE_INPUT_TRM_ROOT = Path("/kaggle/input/medv3-checkpoints/trm")
KAGGLE_WORKING_TRM_ROOT = Path("/kaggle/working/external/TinyRecursiveModels")
LOCAL_TRM_ROOT = Path("data/checkpoints/trm")


@dataclass(slots=True)
class TRMInnerCarry:
    y_state: torch.Tensor
    z_state: torch.Tensor


@dataclass(slots=True)
class TRMCarry:
    inner_carry: TRMInnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: dict[str, torch.Tensor]


@dataclass(slots=True)
class SamsungTRMConfig:
    batch_size: int = 1
    seq_len: int = DEFAULT_TRM_SEQ_LEN
    puzzle_emb_ndim: int = DEFAULT_TRM_HIDDEN_SIZE
    num_puzzle_identifiers: int = DEFAULT_TRM_PUZZLE_IDENTIFIERS
    vocab_size: int = DEFAULT_TRM_VOCAB_SIZE
    H_cycles: int = DEFAULT_TRM_H_CYCLES
    L_cycles: int = DEFAULT_TRM_L_CYCLES
    H_layers: int = 0
    L_layers: int = DEFAULT_TRM_LAYERS
    hidden_size: int = DEFAULT_TRM_HIDDEN_SIZE
    expansion: float = DEFAULT_TRM_EXPANSION
    num_heads: int = DEFAULT_TRM_HEADS
    pos_encodings: str = "rope"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_max_steps: int = DEFAULT_TRM_HALT_MAX_STEPS
    halt_exploration_prob: float = 0.0
    forward_dtype: str = DEFAULT_TRM_FORWARD_DTYPE
    mlp_t: bool = False
    puzzle_emb_len: int = DEFAULT_TRM_PUZZLE_EMB_LEN
    no_ACT_continue: bool = True

    def to_model_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TRMModelBundle:
    model: nn.Module
    config: SamsungTRMConfig
    source: str
    checkpoint_path: Path | None
    repo_root: Path | None
    checkpoint_loaded: bool = False
    medical_compatible: bool = False
    load_diagnostics: list[str] = field(default_factory=list)


class ResidualTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, expansion: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        expanded_size = int(hidden_size * expansion)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, expanded_size),
            nn.GELU(),
            nn.Linear(expanded_size, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(hidden_states)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        hidden_states = hidden_states + attn_output
        ffn_input = self.norm2(hidden_states)
        return hidden_states + self.ffn(ffn_input)


class FallbackTinyRecursiveReasoningModel(nn.Module):
    """TRM-compatible fallback used when official Samsung code is unavailable."""

    def __init__(self, config: SamsungTRMConfig) -> None:
        super().__init__()
        self.config = config
        self.compute_dtype = _resolve_compute_dtype(config.forward_dtype, torch.device("cpu"))

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.seq_len + config.puzzle_emb_len, config.hidden_size)
        self.puzzle_embedding = nn.Embedding(config.num_puzzle_identifiers, config.hidden_size)
        self.puzzle_offsets = nn.Parameter(torch.randn(config.puzzle_emb_len, config.hidden_size) * 0.02)
        self.input_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.recurrent_layers = nn.ModuleList(
            [
                ResidualTransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    expansion=config.expansion,
                )
                for _ in range(config.L_layers)
            ]
        )
        self.latent_norm = nn.LayerNorm(config.hidden_size)
        self.answer_norm = nn.LayerNorm(config.hidden_size)
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.answer_update = nn.Linear(config.hidden_size, config.hidden_size)
        self.latent_update = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.q_head = nn.Linear(config.hidden_size, 2)
        self.y_init = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.z_init = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    @property
    def puzzle_emb(self) -> nn.Embedding:
        return self.puzzle_embedding

    def to(self, *args: Any, **kwargs: Any) -> FallbackTinyRecursiveReasoningModel:
        module = super().to(*args, **kwargs)
        device = next(self.parameters()).device
        self.compute_dtype = _resolve_compute_dtype(self.config.forward_dtype, device)
        return module  # type: ignore[return-value]

    def initial_carry(self, batch: dict[str, torch.Tensor]) -> TRMCarry:
        batch_size = int(batch["inputs"].shape[0])
        total_len = self.config.seq_len + self.config.puzzle_emb_len
        device = batch["inputs"].device
        y_state = self.y_init.expand(batch_size, total_len, self.config.hidden_size).clone().to(device=device, dtype=self.compute_dtype)
        z_state = self.z_init.expand(batch_size, total_len, self.config.hidden_size).clone().to(device=device, dtype=self.compute_dtype)
        return TRMCarry(
            inner_carry=TRMInnerCarry(y_state=y_state, z_state=z_state),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=device),
            halted=torch.ones((batch_size,), dtype=torch.bool, device=device),
            current_data={key: value.clone() for key, value in batch.items()},
        )

    def forward(self, carry: TRMCarry, batch: dict[str, torch.Tensor]) -> tuple[TRMCarry, dict[str, torch.Tensor]]:
        device = batch["inputs"].device
        current_data = {
            key: torch.where(
                carry.halted.view((-1,) + (1,) * (value.ndim - 1)),
                value,
                carry.current_data[key],
            )
            if key in carry.current_data
            else value
            for key, value in batch.items()
        }
        new_steps = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)
        inner_carry = self._reset_carry(carry.halted, carry.inner_carry, batch_size=batch["inputs"].shape[0], device=device)

        sequence_embeddings = self._build_sequence_embeddings(current_data)
        new_inner_carry, logits, q_logits = self._run_recursive_core(inner_carry, sequence_embeddings)
        q_halt_logits = q_logits[..., 0]
        q_continue_logits = q_logits[..., 1]

        new_steps = new_steps + 1
        halted = new_steps >= self.config.halt_max_steps
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }
        return TRMCarry(
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=halted,
            current_data={key: value.clone() for key, value in current_data.items()},
        ), outputs

    def _reset_carry(
        self,
        halted: torch.Tensor,
        inner_carry: TRMInnerCarry,
        batch_size: int,
        device: torch.device,
    ) -> TRMInnerCarry:
        total_len = self.config.seq_len + self.config.puzzle_emb_len
        reset_y = self.y_init.expand(batch_size, total_len, self.config.hidden_size).to(device=device, dtype=self.compute_dtype)
        reset_z = self.z_init.expand(batch_size, total_len, self.config.hidden_size).to(device=device, dtype=self.compute_dtype)
        halted_mask = halted.view(batch_size, 1, 1)
        y_state = torch.where(halted_mask, reset_y, inner_carry.y_state)
        z_state = torch.where(halted_mask, reset_z, inner_carry.z_state)
        return TRMInnerCarry(y_state=y_state, z_state=z_state)

    def _build_sequence_embeddings(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = batch["inputs"]
        puzzle_identifiers = batch["puzzle_identifiers"].clamp(min=0, max=self.config.num_puzzle_identifiers - 1)
        token_embeddings = self.token_embedding(inputs)
        puzzle_embeddings = self.puzzle_embedding(puzzle_identifiers).unsqueeze(1)
        puzzle_embeddings = puzzle_embeddings + self.puzzle_offsets.unsqueeze(0)
        sequence_embeddings = torch.cat((puzzle_embeddings, token_embeddings), dim=1)
        position_ids = torch.arange(sequence_embeddings.size(1), device=inputs.device)
        position_embeddings = self.position_embedding(position_ids).unsqueeze(0)
        sequence_embeddings = self.input_projection(sequence_embeddings + position_embeddings)
        return sequence_embeddings.to(dtype=self.compute_dtype)

    def _run_recursive_core(
        self,
        carry: TRMInnerCarry,
        sequence_embeddings: torch.Tensor,
    ) -> tuple[TRMInnerCarry, torch.Tensor, torch.Tensor]:
        y_state = carry.y_state
        z_state = carry.z_state
        hidden_states = sequence_embeddings

        for _ in range(self.config.H_cycles):
            latent_states = z_state + hidden_states
            for _ in range(self.config.L_cycles):
                for layer in self.recurrent_layers:
                    latent_states = layer(latent_states)
            z_state = self.latent_norm(z_state + torch.tanh(self.latent_update(latent_states)))
            y_state = self.answer_norm(y_state + torch.tanh(self.answer_update(z_state + sequence_embeddings)))
            hidden_states = y_state

        final_states = self.output_norm(hidden_states)
        answer_states = final_states[:, self.config.puzzle_emb_len :, :]
        logits = self.lm_head(answer_states)
        q_logits = self.q_head(final_states[:, 0, :])
        return TRMInnerCarry(y_state=y_state, z_state=z_state), logits, q_logits


def stablemax_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = DEFAULT_IGNORE_LABEL_ID,
) -> torch.Tensor:
    if logits.ndim < 2:
        raise ValueError("logits must have at least 2 dimensions.")
    logits = logits.float()
    targets = targets.long()
    log_partition = torch.logsumexp(logits, dim=-1)
    gathered_logits = torch.gather(logits, dim=-1, index=targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    losses = log_partition - gathered_logits
    valid_mask = targets != ignore_index
    if not torch.any(valid_mask):
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    return losses[valid_mask].mean()


def get_trm_paths(path_hint: str | Path | None = None) -> tuple[Path | None, Path | None, str]:
    candidates: list[tuple[Path, str]] = []
    seen: set[Path] = set()

    def register(path: Path, source: str) -> None:
        resolved = path.resolve(strict=False)
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append((path, source))

    if path_hint is not None:
        raw_hint = Path(path_hint)
        resolved_hint = raw_hint if raw_hint.is_absolute() else KaggleEnv.path(raw_hint)
        register(resolved_hint, "path_hint")

    resolved_checkpoint = KagglePathResolver.resolve_trm_checkpoint()
    resolved_repo = KagglePathResolver.resolve_trm_repo()
    if resolved_checkpoint is not None:
        register(Path(resolved_checkpoint), "resolver_checkpoint")
    if resolved_repo is not None:
        register(Path(resolved_repo), "resolver_repo")

    register(KAGGLE_INPUT_TRM_ROOT, "kaggle_input")
    register(KAGGLE_WORKING_TRM_ROOT, "kaggle_working")
    register(KaggleEnv.path(DEFAULT_EXTERNAL_TRM_REPO), "workspace_external")
    register(KaggleEnv.path(LOCAL_TRM_ROOT), "local_data")
    register(DEFAULT_EXTERNAL_TRM_REPO, "repo_default")

    for candidate, source in candidates:
        if not candidate.exists():
            continue
        if candidate.is_file():
            repo_root = _resolve_repo_root(candidate)
            return repo_root, candidate, source

        repo_root = _resolve_repo_root(candidate)
        checkpoint_path = _resolve_preferred_checkpoint(repo_root) if repo_root is not None else _resolve_checkpoint_path(candidate)
        if repo_root is None and checkpoint_path is not None:
            repo_root = _resolve_repo_root(checkpoint_path)
        if repo_root is not None and checkpoint_path is None:
            checkpoint_path = _resolve_preferred_checkpoint(repo_root)
        if repo_root is not None or checkpoint_path is not None:
            return repo_root, checkpoint_path, source

    return None, None, "unresolved"


def verify_trm_repo(repo_root: Path, checkpoint_path: Path | None = None) -> tuple[bool, list[str]]:
    diagnostics: list[str] = []
    repo_exists = repo_root.exists()
    samsung_source = repo_root / OFFICIAL_TRM_SOURCE_FILE
    legacy_source = repo_root / OFFICIAL_TRM_ALT_SOURCE_FILE
    config_path = repo_root / OFFICIAL_TRM_CONFIG_PATH
    checkpoint_target = checkpoint_path if checkpoint_path is not None else repo_root / "checkpoints"

    samsung_exists = samsung_source.exists()
    legacy_exists = legacy_source.exists()
    config_exists = config_path.exists()
    checkpoint_exists = checkpoint_target.exists()

    if samsung_exists:
        structure_message = "TRM repo: SAMSUNG structure (models/recursive_reasoning/trm.py)"
        logger.info(structure_message)
    elif legacy_exists:
        structure_message = "TRM repo: LEGACY structure (src/trm.py)"
        logger.info(structure_message)
    else:
        structure_message = (
            "TRM repo: UNKNOWN structure, neither models/recursive_reasoning/trm.py nor src/trm.py found"
        )
        logger.warning(structure_message)
    diagnostics.append(structure_message)

    checks: list[tuple[str, bool]] = [
        ("repo_root_exists", repo_exists),
        (str(OFFICIAL_TRM_SOURCE_FILE).replace("\\", "/"), samsung_exists),
        (str(OFFICIAL_TRM_CONFIG_PATH).replace("\\", "/"), config_exists),
        (str(checkpoint_target), checkpoint_exists),
    ]
    if legacy_exists:
        legacy_label = str(OFFICIAL_TRM_ALT_SOURCE_FILE).replace("\\", "/")
        checks.append((f"{legacy_label} (legacy)", True))

    all_exist = True
    for label, exists in checks:
        status = "EXISTS" if exists else "MISSING"
        message = f"TRM check: {label} -> {status}"
        diagnostics.append(message)
        logger.info(message)
        all_exist = all_exist and exists
    return all_exist, diagnostics


def adapt_samsung_checkpoint(raw_checkpoint: Any) -> tuple[dict[str, Any], list[str]]:
    diagnostics: list[str] = []
    if not isinstance(raw_checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(raw_checkpoint).__name__}")

    top_level_keys = list(raw_checkpoint.keys())
    if top_level_keys:
        preview = top_level_keys[:20]
        message = f"Raw checkpoint keys ({len(top_level_keys)}): {preview}"
    else:
        message = "Raw checkpoint keys (0): []"
    diagnostics.append(message)
    logger.info(message)

    adapted: dict[str, Any] = {}
    for key in ("config", "cfg", "model_config", "hyper_parameters", "arch"):
        if key in raw_checkpoint:
            adapted[key] = raw_checkpoint[key]

    state_dict: dict[str, torch.Tensor] | None = None
    for key in ("ema_state_dict", "ema", "model_state_dict", "state_dict", "model"):
        value = raw_checkpoint.get(key)
        if isinstance(value, dict) and all(isinstance(tensor, torch.Tensor) for tensor in value.values()):
            state_dict = value
            diagnostics.append(f"Checkpoint state_dict source: {key}")
            break
    if state_dict is None and all(isinstance(value, torch.Tensor) for value in raw_checkpoint.values()):
        state_dict = raw_checkpoint  # type: ignore[assignment]
        diagnostics.append("Checkpoint state_dict source: top_level_tensor_dict")
    if state_dict is None:
        raise ValueError("Could not locate a tensor state_dict in Samsung checkpoint payload.")

    adapted["model_state_dict"] = state_dict
    return adapted, diagnostics


def load_trm_config(repo_root: str | Path) -> dict[str, Any]:
    """Load Samsung TRM config from config/arch/trm.yaml when available."""
    resolved_repo_root = Path(repo_root)
    if not resolved_repo_root.exists():
        logger.warning("TRM repo root not found while loading config: %s", resolved_repo_root)
        return {}
    config_payload = _load_config_from_repo(resolved_repo_root)
    if config_payload:
        logger.info("Loaded TRM config payload keys: %s", sorted(config_payload.keys()))
    return config_payload


def inspect_checkpoint(
    checkpoint_path: str | Path,
    checkpoint_payload: Any | None = None,
) -> tuple[Any, dict[str, Any], list[str]]:
    """Inspect checkpoint payload and emit compact structure diagnostics."""
    diagnostics: list[str] = []
    checkpoint = Path(checkpoint_path)
    payload = checkpoint_payload if checkpoint_payload is not None else torch.load(checkpoint, map_location="cpu")

    top_level_keys = list(payload.keys()) if isinstance(payload, dict) else []
    if isinstance(payload, dict):
        state_like = _extract_state_dict(payload)
    else:
        state_like = None
    if state_like is None and isinstance(payload, dict) and all(isinstance(value, torch.Tensor) for value in payload.values()):
        state_like = payload  # type: ignore[assignment]

    sample_keys = list(state_like.keys())[:10] if isinstance(state_like, dict) else []
    sample_shapes: dict[str, tuple[int, ...]] = {}
    if isinstance(state_like, dict):
        for key, value in list(state_like.items())[:5]:
            if isinstance(value, torch.Tensor):
                sample_shapes[key] = tuple(value.shape)

    info = {
        "checkpoint_path": str(checkpoint),
        "payload_type": type(payload).__name__,
        "top_level_key_count": len(top_level_keys),
        "top_level_keys_preview": top_level_keys[:20],
        "has_state_dict": isinstance(payload, dict) and "state_dict" in payload,
        "has_model_state_dict": isinstance(payload, dict) and "model_state_dict" in payload,
        "has_model": isinstance(payload, dict) and "model" in payload,
        "tensor_count": len(state_like) if isinstance(state_like, dict) else 0,
        "sample_keys": sample_keys,
        "sample_shapes": sample_shapes,
    }

    diagnostics.append(f"Checkpoint inspect: {json.dumps(info, ensure_ascii=True)}")
    logger.info("Checkpoint inspect: %s", json.dumps(info, ensure_ascii=True))
    return payload, info, diagnostics


def normalize_state_dict_for_model(
    raw_state_dict: dict[str, torch.Tensor],
    model: nn.Module,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Map checkpoint tensor keys to model keys with prefix normalization."""
    diagnostics: list[str] = []
    model_state = model.state_dict()
    model_keys = set(model_state.keys())

    mapped: dict[str, torch.Tensor] = {}
    unmappable: list[str] = []
    duplicate_targets: list[str] = []
    shape_mismatches: list[str] = []

    for key, value in raw_state_dict.items():
        target_key: str | None = None
        for candidate in _state_dict_key_candidates(key):
            target_tensor = model_state.get(candidate)
            if target_tensor is None:
                continue
            if tuple(target_tensor.shape) != tuple(value.shape):
                shape_mismatches.append(f"{key} -> {candidate}")
                continue
            target_key = candidate
            break
        if target_key is None:
            unmappable.append(key)
            continue
        if target_key in mapped:
            duplicate_targets.append(f"{key} -> {target_key}")
            continue
        mapped[target_key] = value

    missing = sorted(model_keys.difference(mapped.keys()))
    unexpected = sorted(set(raw_state_dict.keys()).difference(set(mapped.keys())))
    diagnostics.append(
        f"State dict compatibility: mapped={len(mapped)}/{len(raw_state_dict)}, "
        f"unexpected={len(unexpected)}, missing={len(missing)}, "
        f"shape_mismatch={len(shape_mismatches)}, duplicate_targets={len(duplicate_targets)}"
    )
    if unexpected:
        diagnostics.append(f"Unexpected keys preview: {unexpected[:10]}")
    if missing:
        diagnostics.append(f"Missing keys preview: {missing[:10]}")
    if shape_mismatches:
        diagnostics.append(f"Shape mismatch preview: {shape_mismatches[:10]}")
    if unmappable:
        diagnostics.append(f"Unmappable keys preview: {unmappable[:10]}")
    for note in diagnostics:
        logger.info(note)
    return mapped, diagnostics


def create_dummy_input_for_trm(
    config: SamsungTRMConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch_size = 1
    seq_len = max(int(config.seq_len), 1)
    vocab_size = max(int(config.vocab_size), 1)
    puzzle_count = max(int(config.num_puzzle_identifiers), 1)
    dummy_inputs = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.int64, device=device)
    dummy_labels = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.int64, device=device)
    dummy_puzzles = torch.zeros((batch_size,), dtype=torch.int64, device=device).clamp(max=puzzle_count - 1)
    return {
        "inputs": dummy_inputs,
        "labels": dummy_labels,
        "puzzle_identifiers": dummy_puzzles,
    }


def load_trm_model_bundle(
    model_path: str | Path,
    device: str | torch.device = "cpu",
    seq_len: int = DEFAULT_TRM_SEQ_LEN,
    num_puzzle_identifiers: int = DEFAULT_TRM_PUZZLE_IDENTIFIERS,
    forward_dtype: str = DEFAULT_TRM_FORWARD_DTYPE,
    allow_official_untrained: bool = False,
    strict_validate: bool = True,
) -> TRMModelBundle:
    T4Hardening.setup_memory()
    T4Hardening.verify_fp16(forward_dtype)
    resolved_device = torch.device(device)
    raw_path = Path(model_path)
    path = raw_path if raw_path.is_absolute() else KaggleEnv.path(raw_path)
    diagnostics: list[str] = []
    repo_root, checkpoint_path, path_source = get_trm_paths(path)
    diagnostics.append(f"TRM path source selected: {path_source}")
    if path_source == "unresolved":
        diagnostics.append(f"Unable to resolve TRM from provided path hint: {path}")

    if repo_root is None:
        repo_root = _resolve_repo_root(path)
    if checkpoint_path is None:
        checkpoint_path = _resolve_checkpoint_path(path)
    if repo_root is None or checkpoint_path is None:
        inferred_repo_root = _resolve_external_repo_root(path)
        if inferred_repo_root is not None:
            repo_root = repo_root or inferred_repo_root
            checkpoint_path = checkpoint_path or _resolve_preferred_checkpoint(inferred_repo_root)
    if repo_root is not None:
        repo_ok, repo_diagnostics = verify_trm_repo(repo_root, checkpoint_path=checkpoint_path)
        diagnostics.extend(repo_diagnostics)
        if not repo_ok:
            diagnostics.append(f"TRM repository structure check failed for {repo_root}")
        if checkpoint_path is None:
            checkpoint_path = _resolve_preferred_checkpoint(repo_root)

    if repo_root is not None and checkpoint_path is not None and checkpoint_path.exists():
        try:
            bundle = load_real_trm(
                checkpoint_path=checkpoint_path,
                repo_root=repo_root,
                device=resolved_device,
                seq_len=seq_len,
                num_puzzle_identifiers=num_puzzle_identifiers,
                forward_dtype=forward_dtype,
                strict_validate=strict_validate,
            )
            bundle.load_diagnostics = [*diagnostics, *bundle.load_diagnostics]
            return bundle
        except Exception as exc:
            message = f"Unable to load official Samsung TRM. Falling back to internal wrapper: {exc}"
            diagnostics.append(message)
            diagnostics.append(traceback.format_exc())
            logger.error(message)
            logger.error("TRM load traceback:\n%s", diagnostics[-1])
    elif checkpoint_path is not None and not checkpoint_path.exists():
        diagnostics.append(f"Resolved TRM checkpoint path does not exist: {checkpoint_path}")

    if repo_root is not None and allow_official_untrained:
        try:
            return initialize_official_medical_trm(
                repo_root=repo_root,
                device=resolved_device,
                seq_len=seq_len,
                num_puzzle_identifiers=num_puzzle_identifiers,
                forward_dtype=forward_dtype,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            message = f"Unable to initialize untrained official Samsung TRM. Falling back to internal wrapper: {exc}"
            diagnostics.append(message)
            logger.warning(message)

    checkpoint_payload = _load_checkpoint_payload(checkpoint_path) if checkpoint_path is not None else None
    config = _build_config(
        checkpoint_payload=checkpoint_payload,
        repo_root=repo_root,
        checkpoint_path=checkpoint_path,
        seq_len=seq_len,
        num_puzzle_identifiers=num_puzzle_identifiers,
        forward_dtype=forward_dtype,
        force_medical_defaults=True,
    )

    model: nn.Module = FallbackTinyRecursiveReasoningModel(config)
    source = "fallback"
    checkpoint_loaded = False

    if checkpoint_payload is not None:
        state_dict = _extract_state_dict(checkpoint_payload)
        if state_dict is not None:
            normalized_state_dict = _normalize_state_dict_keys(state_dict)
            compatible_state_dict, compatibility_notes = _select_compatible_state_dict(model, normalized_state_dict)
            diagnostics.extend(compatibility_notes)
            incompatible = model.load_state_dict(compatible_state_dict, strict=False) if compatible_state_dict else None
            checkpoint_loaded = bool(compatible_state_dict)
            if incompatible is not None and (incompatible.missing_keys or incompatible.unexpected_keys):
                diagnostics.append(
                    f"Fallback TRM checkpoint was partial: missing={len(incompatible.missing_keys)} "
                    f"unexpected={len(incompatible.unexpected_keys)}"
                )
            if not compatible_state_dict:
                diagnostics.append("Checkpoint state_dict had no compatible tensors for the internal fallback TRM.")

    model = model.to(resolved_device)
    model.eval()
    return TRMModelBundle(
        model=model,
        config=config,
        source=source,
        checkpoint_path=checkpoint_path,
        repo_root=repo_root,
        checkpoint_loaded=checkpoint_loaded,
        medical_compatible=checkpoint_loaded and _is_medical_compatible_config(config),
        load_diagnostics=diagnostics,
    )


def load_real_trm(
    checkpoint_path: str | Path,
    repo_root: str | Path | None = None,
    device: str | torch.device = "cpu",
    seq_len: int = DEFAULT_TRM_SEQ_LEN,
    num_puzzle_identifiers: int = DEFAULT_TRM_PUZZLE_IDENTIFIERS,
    forward_dtype: str = DEFAULT_TRM_FORWARD_DTYPE,
    strict_validate: bool = True,
) -> TRMModelBundle:
    T4Hardening.setup_memory()
    T4Hardening.verify_fp16(forward_dtype)
    resolved_device = torch.device(device)
    diagnostics: list[str] = []
    raw_checkpoint = Path(checkpoint_path)
    checkpoint = raw_checkpoint if raw_checkpoint.is_absolute() else KaggleEnv.path(raw_checkpoint)
    logger.info("=== TRM Load Attempt ===")
    logger.info(f"Attempting to load TRM from: {checkpoint}")
    logger.info(f"TRM checkpoint exists: {checkpoint.exists()}")
    diagnostics.append("=== TRM Load Attempt ===")

    if repo_root is not None:
        raw_repo_root = Path(repo_root)
        resolved_repo_root = raw_repo_root if raw_repo_root.is_absolute() else KaggleEnv.path(raw_repo_root)
    else:
        resolved_repo_root = _resolve_repo_root(checkpoint)
    logger.info(f"Resolved TRM repo root: {resolved_repo_root}")
    diagnostics.append(f"Resolved checkpoint path: {checkpoint}")
    diagnostics.append(f"Resolved repo root: {resolved_repo_root}")

    if not checkpoint.exists():
        raise FileNotFoundError(f"Samsung TRM checkpoint not found: {checkpoint}")
    if resolved_repo_root is None:
        raise FileNotFoundError("Samsung TRM repository root could not be resolved.")
    try:
        checkpoint_size = int(checkpoint.stat().st_size)
        logger.info("TRM checkpoint file size bytes: %s", checkpoint_size)
        diagnostics.append(f"Checkpoint size bytes: {checkpoint_size}")
    except OSError as exc:
        logger.warning("Unable to read checkpoint size for %s: %s", checkpoint, exc)
        diagnostics.append(f"Checkpoint size unavailable: {exc}")

    sample_python_files: list[str] = []
    try:
        sample_python_files = [str(path) for path in resolved_repo_root.glob("**/*.py")][:5]
    except OSError:
        sample_python_files = []
    logger.info("TRM repo sample files: %s", sample_python_files)
    diagnostics.append(f"TRM repo sample python files: {sample_python_files}")

    repo_ok, repo_diagnostics = verify_trm_repo(resolved_repo_root, checkpoint_path=checkpoint)
    diagnostics.extend(repo_diagnostics)
    if not repo_ok:
        logger.warning("TRM repo verification found missing expected files under %s", resolved_repo_root)

    try:
        official_class = _import_samsung_trm(resolved_repo_root)
        if official_class is None:
            raise ImportError(f"Official Samsung TRM class '{OFFICIAL_TRM_CLASS_NAME}' could not be imported from {resolved_repo_root}.")
        logger.info(f"TRM module imported successfully from: {resolved_repo_root}")
        diagnostics.append(f"TRM class resolved: {official_class.__name__}")

        raw_checkpoint_payload = torch.load(checkpoint, map_location="cpu")
        logger.info("Checkpoint loaded from disk successfully.")
        diagnostics.append("Checkpoint payload loaded from disk.")
        _checkpoint_payload, _checkpoint_info, checkpoint_inspect_diagnostics = inspect_checkpoint(
            checkpoint_path=checkpoint,
            checkpoint_payload=raw_checkpoint_payload,
        )
        diagnostics.extend(checkpoint_inspect_diagnostics)
        adapted_checkpoint, checkpoint_diagnostics = adapt_samsung_checkpoint(raw_checkpoint_payload)
        diagnostics.extend(checkpoint_diagnostics)

        state_dict = _extract_state_dict(adapted_checkpoint)
        if state_dict is None:
            raise ValueError("Adapted checkpoint payload does not include a usable state_dict.")
        prefix_normalized_state_dict = _normalize_state_dict_keys(state_dict)
        diagnostics.append(f"Normalized checkpoint tensor count: {len(prefix_normalized_state_dict)}")
        logger.info("Checkpoint tensors normalized: %s keys", len(prefix_normalized_state_dict))

        config = _build_config(
            checkpoint_payload=adapted_checkpoint,
            repo_root=resolved_repo_root,
            checkpoint_path=checkpoint,
            seq_len=seq_len,
            num_puzzle_identifiers=num_puzzle_identifiers,
            forward_dtype=forward_dtype,
            force_medical_defaults=False,
            inferred_state_dict=prefix_normalized_state_dict,
        )
        repo_config = load_trm_config(resolved_repo_root)
        if repo_config:
            diagnostics.append(f"Loaded TRM yaml config keys: {sorted(repo_config.keys())}")

        logger.info("Initializing official TRM model with inferred config.")
        diagnostics.append(
            "Model config summary: "
            f"seq_len={config.seq_len}, vocab_size={config.vocab_size}, "
            f"num_puzzle_identifiers={config.num_puzzle_identifiers}, hidden_size={config.hidden_size}"
        )
        model = official_class(config.to_model_dict())
        mapped_state_dict, state_mapping_diagnostics = normalize_state_dict_for_model(
            raw_state_dict=prefix_normalized_state_dict,
            model=model,
        )
        diagnostics.extend(state_mapping_diagnostics)
        if not mapped_state_dict:
            message = "Checkpoint mapping produced zero compatible tensors for official Samsung TRM."
            if strict_validate:
                raise RuntimeError(message)
            logger.warning(message)
            diagnostics.append(f"WARNING: {message}")
        strict_state_load = len(mapped_state_dict) == len(model.state_dict())
        diagnostics.append(f"State dict load mode: {'strict' if strict_state_load else 'non_strict'}")
        incompatible = model.load_state_dict(mapped_state_dict, strict=strict_state_load)
        if incompatible.missing_keys:
            diagnostics.append(f"State dict missing keys after load (preview): {incompatible.missing_keys[:10]}")
        if incompatible.unexpected_keys:
            diagnostics.append(f"State dict unexpected keys after load (preview): {incompatible.unexpected_keys[:10]}")
        logger.info(
            "TRM weights load completed with mapped tensors=%s, strict=%s",
            len(mapped_state_dict),
            strict_state_load,
        )
        diagnostics.append(f"TRM mapped tensor count loaded: {len(mapped_state_dict)}")
    except Exception as exc:
        logger.error(f"TRM load failed: {exc}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

    model = model.to(resolved_device)
    if hasattr(model, "config") and hasattr(model.config, "forward_dtype"):
        try:
            model.config.forward_dtype = "float16"
        except Exception:
            pass
    model.eval()
    validation_ok, validation_details = _validate_loaded_trm(
        model=model,
        config=config,
        device=resolved_device,
    )
    diagnostics.extend(validation_details)
    if not validation_ok:
        if strict_validate:
            raise RuntimeError("Loaded TRM checkpoint failed dummy forward-pass validation.")
        logger.warning("TRM dummy forward-pass validation failed, but strict_validate=False so load will continue.")
        diagnostics.append("WARNING: Validation failed but strict_validate=False; continuing with loaded model.")
    else:
        logger.info("TRM dummy forward-pass validation succeeded.")
    medical_compatible = _is_medical_compatible_config(config)
    if not medical_compatible:
        warning_message = "Loaded ARC-style checkpoint. Marking medical_compatible=False and enabling input remap downstream."
        logger.warning(warning_message)
        diagnostics.append(f"WARNING: {warning_message}")
    setattr(model, "medical_compatible", medical_compatible)
    setattr(model, "input_remap_active", not medical_compatible)
    return TRMModelBundle(
        model=model,
        config=config,
        source="official",
        checkpoint_path=checkpoint,
        repo_root=resolved_repo_root,
        checkpoint_loaded=validation_ok or not strict_validate,
        medical_compatible=medical_compatible,
        load_diagnostics=diagnostics,
    )


def _validate_loaded_trm(
    *,
    model: nn.Module,
    config: SamsungTRMConfig,
    device: torch.device,
) -> tuple[bool, list[str]]:
    diagnostics: list[str] = []
    dummy_batch = create_dummy_input_for_trm(config=config, device=device)
    shape_summary = {key: tuple(value.shape) for key, value in dummy_batch.items()}
    diagnostics.append(
        "Running TRM dummy forward validation with "
        f"seq_len={config.seq_len}, vocab_size={config.vocab_size}, "
        f"puzzle_count={config.num_puzzle_identifiers}"
    )
    diagnostics.append(f"Dummy input shapes: {shape_summary}")
    try:
        outputs: Any | None = None
        validation_api = "uninitialized"
        with torch.no_grad():
            if hasattr(model, "initial_carry"):
                carry = model.initial_carry(dummy_batch)  # type: ignore[call-arg]
                validation_api = "initial_carry+forward(carry,batch)"
                call_output = model(carry, dummy_batch)
                if (
                    isinstance(call_output, (tuple, list))
                    and len(call_output) == 2
                    and isinstance(call_output[1], dict)
                ):
                    outputs = call_output[1]
                else:
                    outputs = call_output
            if outputs is None:
                try:
                    validation_api = "forward(**batch)"
                    outputs = model(**dummy_batch)  # type: ignore[misc]
                except Exception:
                    validation_api = "forward(batch_dict)"
                    outputs = model(dummy_batch)  # type: ignore[call-arg]
        diagnostics.append(f"Validation API: {validation_api}")
        diagnostics.append(f"Validation output type: {type(outputs).__name__}")

        output_payload = _extract_validation_output_payload(outputs)
        if not isinstance(output_payload, dict):
            diagnostics.append("Dummy forward output is not a dict payload with expected TRM keys.")
            return False, diagnostics
        required_keys = {"logits", "q_halt_logits", "q_continue_logits"}
        missing = sorted(required_keys.difference(output_payload.keys()))
        if missing:
            diagnostics.append(f"Dummy forward output missing required keys: {missing}")
            return False, diagnostics
        logits = output_payload["logits"]
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            diagnostics.append("Dummy forward logits tensor has unexpected shape/type.")
            return False, diagnostics
        diagnostics.append(
            "Dummy forward validation passed with logits shape "
            f"{tuple(logits.shape)} and dtype={str(logits.dtype)}"
        )
        return True, diagnostics
    except Exception as exc:
        diagnostics.append(f"Dummy forward validation failed: {exc}")
        diagnostics.append(traceback.format_exc())
        return False, diagnostics


def _extract_validation_output_payload(outputs: Any) -> dict[str, Any] | None:
    if isinstance(outputs, dict):
        return outputs
    if isinstance(outputs, (tuple, list)):
        for item in outputs:
            if isinstance(item, dict):
                return item
    return None


def initialize_official_medical_trm(
    repo_root: str | Path,
    device: str | torch.device = "cpu",
    seq_len: int = DEFAULT_TRM_SEQ_LEN,
    num_puzzle_identifiers: int = DEFAULT_TRM_PUZZLE_IDENTIFIERS,
    forward_dtype: str = DEFAULT_TRM_FORWARD_DTYPE,
    diagnostics: list[str] | None = None,
) -> TRMModelBundle:
    T4Hardening.setup_memory()
    T4Hardening.verify_fp16(forward_dtype)
    resolved_device = torch.device(device)
    raw_repo_root = Path(repo_root)
    resolved_repo_root = raw_repo_root if raw_repo_root.is_absolute() else KaggleEnv.path(raw_repo_root)
    if not resolved_repo_root.exists():
        raise FileNotFoundError(f"Samsung TRM repository root not found: {resolved_repo_root}")

    official_class = _import_samsung_trm(resolved_repo_root)
    if official_class is None:
        raise ImportError(f"Official Samsung TRM class '{OFFICIAL_TRM_CLASS_NAME}' could not be imported from {resolved_repo_root}.")

    config = _build_config(
        checkpoint_payload=None,
        repo_root=resolved_repo_root,
        checkpoint_path=None,
        seq_len=seq_len,
        num_puzzle_identifiers=num_puzzle_identifiers,
        forward_dtype=forward_dtype,
        force_medical_defaults=True,
    )
    model = official_class(config.to_model_dict()).to(resolved_device)
    if hasattr(model, "config") and hasattr(model.config, "forward_dtype"):
        try:
            model.config.forward_dtype = "float16"
        except Exception:
            pass
    model.train()
    return TRMModelBundle(
        model=model,
        config=config,
        source="official_untrained",
        checkpoint_path=None,
        repo_root=resolved_repo_root,
        checkpoint_loaded=False,
        medical_compatible=_is_medical_compatible_config(config),
        load_diagnostics=list(diagnostics or []) + ["Initialized official Samsung TRM core from scratch for medical fine-tuning."],
    )


def _build_config(
    checkpoint_payload: dict[str, Any] | None,
    repo_root: Path | None,
    checkpoint_path: Path | None,
    seq_len: int,
    num_puzzle_identifiers: int,
    forward_dtype: str,
    force_medical_defaults: bool = True,
    inferred_state_dict: dict[str, torch.Tensor] | None = None,
) -> SamsungTRMConfig:
    config_payload: dict[str, Any] = {
        "seq_len": seq_len,
        "num_puzzle_identifiers": num_puzzle_identifiers,
        "forward_dtype": forward_dtype,
    }
    if repo_root is not None:
        config_payload.update(_load_config_from_repo(repo_root))
    if checkpoint_path is not None:
        config_payload.update(_load_checkpoint_sidecar_config(checkpoint_path))
    if checkpoint_payload is not None:
        config_payload.update(_extract_config_payload(checkpoint_payload))
    if inferred_state_dict is not None:
        config_payload.update(_infer_config_from_state_dict(inferred_state_dict))

    config_payload["seq_len"] = seq_len
    if force_medical_defaults:
        config_payload["vocab_size"] = DEFAULT_TRM_VOCAB_SIZE
        config_payload["num_puzzle_identifiers"] = num_puzzle_identifiers
        config_payload["H_cycles"] = DEFAULT_TRM_H_CYCLES
        config_payload["L_cycles"] = DEFAULT_TRM_L_CYCLES
        config_payload["H_layers"] = 0
        config_payload["L_layers"] = DEFAULT_TRM_LAYERS
        config_payload["hidden_size"] = DEFAULT_TRM_HIDDEN_SIZE
        config_payload["puzzle_emb_ndim"] = DEFAULT_TRM_HIDDEN_SIZE
        config_payload["num_heads"] = DEFAULT_TRM_HEADS
        config_payload["puzzle_emb_len"] = DEFAULT_TRM_PUZZLE_EMB_LEN
    else:
        config_payload.setdefault("vocab_size", DEFAULT_TRM_VOCAB_SIZE)
        config_payload.setdefault("num_puzzle_identifiers", num_puzzle_identifiers)
        config_payload.setdefault("H_cycles", DEFAULT_TRM_H_CYCLES)
        config_payload.setdefault("L_cycles", DEFAULT_TRM_L_CYCLES)
        config_payload.setdefault("H_layers", 0)
        config_payload.setdefault("L_layers", DEFAULT_TRM_LAYERS)
        config_payload.setdefault("hidden_size", DEFAULT_TRM_HIDDEN_SIZE)
        config_payload.setdefault("puzzle_emb_ndim", int(config_payload.get("hidden_size", DEFAULT_TRM_HIDDEN_SIZE)))
        config_payload.setdefault("num_heads", DEFAULT_TRM_HEADS)
        config_payload.setdefault("puzzle_emb_len", DEFAULT_TRM_PUZZLE_EMB_LEN)
    config_payload["batch_size"] = int(config_payload.get("batch_size", 1))
    requested_dtype = str(config_payload.get("forward_dtype", forward_dtype)).lower()
    if requested_dtype in {"bfloat16", "bf16"}:
        logger.warning("Requested TRM forward dtype %s is incompatible with T4. Forcing float16.", requested_dtype)
    config_payload["forward_dtype"] = "float16" if requested_dtype in {"bfloat16", "bf16", "float16", "fp16"} else requested_dtype
    return SamsungTRMConfig(**config_payload)


def _load_config_from_repo(repo_root: Path) -> dict[str, Any]:
    config_path = repo_root / OFFICIAL_TRM_CONFIG_PATH
    if not config_path.exists():
        logger.warning("TRM config not found at %s; using defaults.", config_path)
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    logger.info("Loaded TRM config from %s", config_path)
    payload = {key: value for key, value in payload.items() if key != "name" and key != "loss"}
    return _resolve_simple_interpolations(payload)


def _load_checkpoint_sidecar_config(checkpoint_path: Path) -> dict[str, Any]:
    for config_name in CHECKPOINT_CONFIG_NAMES:
        candidate = checkpoint_path.parent / config_name
        if not candidate.exists():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
        except Exception as exc:
            logger.warning("Unable to load TRM sidecar config %s: %s", candidate, exc)
            continue
        if isinstance(payload, dict) and isinstance(payload.get("arch"), dict):
            arch_payload = {key: value for key, value in dict(payload["arch"]).items() if key not in {"name", "loss"}}
            return _resolve_simple_interpolations(arch_payload)
        if isinstance(payload, dict):
            normalized_payload = {key: value for key, value in dict(payload).items() if key not in {"name", "loss", "arch"}}
            return _resolve_simple_interpolations(normalized_payload)
    return {}


def _extract_config_payload(checkpoint_payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("config", "cfg", "model_config", "hyper_parameters", "arch"):
        value = checkpoint_payload.get(key)
        if isinstance(value, SamsungTRMConfig):
            return value.to_model_dict()
        if isinstance(value, dict):
            return dict(value)
    return {}


def _load_checkpoint_payload(checkpoint_path: Path) -> dict[str, Any] | None:
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        logger.warning("Unable to load TRM checkpoint %s: %s", checkpoint_path, exc)
        return None
    if isinstance(payload, dict):
        return payload
    return {"state_dict": payload}


def _extract_state_dict(checkpoint_payload: dict[str, Any]) -> dict[str, torch.Tensor] | None:
    for key in ("ema_state_dict", "ema", "model_state_dict", "state_dict", "model"):
        value = checkpoint_payload.get(key)
        if isinstance(value, dict) and all(isinstance(tensor, torch.Tensor) for tensor in value.values()):
            return value
    if all(isinstance(value, torch.Tensor) for value in checkpoint_payload.values()):
        return checkpoint_payload  # type: ignore[return-value]
    return None


def _normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized_state_dict: dict[str, torch.Tensor] = {}
    prefixes = ("_orig_mod.", "module.", "model.", "ema_model.")
    for key, value in state_dict.items():
        normalized_key = key
        prefix_removed = True
        while prefix_removed:
            prefix_removed = False
            for prefix in prefixes:
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
                    prefix_removed = True
        normalized_state_dict[normalized_key] = value
    return normalized_state_dict


def _state_dict_key_candidates(key: str) -> tuple[str, ...]:
    candidates = [
        key,
        key.replace("model.", "", 1) if key.startswith("model.") else key,
        key.replace("net.", "", 1) if key.startswith("net.") else key,
        key.replace("trm.", "", 1) if key.startswith("trm.") else key,
    ]
    if not key.startswith("model."):
        candidates.append(f"model.{key}")
    deduplicated: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduplicated.append(candidate)
    return tuple(deduplicated)


def _infer_config_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    inferred: dict[str, Any] = {}

    token_embedding = state_dict.get("inner.embed_tokens.embedding_weight")
    if token_embedding is None:
        token_embedding = state_dict.get("embed_tokens.embedding_weight")
    if isinstance(token_embedding, torch.Tensor) and token_embedding.ndim == 2:
        inferred["vocab_size"] = int(token_embedding.shape[0])
        inferred["hidden_size"] = int(token_embedding.shape[1])

    lm_head = state_dict.get("inner.lm_head.weight")
    if lm_head is None:
        lm_head = state_dict.get("lm_head.weight")
    if isinstance(lm_head, torch.Tensor) and lm_head.ndim == 2:
        inferred.setdefault("vocab_size", int(lm_head.shape[0]))
        inferred.setdefault("hidden_size", int(lm_head.shape[1]))

    puzzle_embedding = state_dict.get("inner.puzzle_emb.weights")
    if puzzle_embedding is None:
        puzzle_embedding = state_dict.get("puzzle_emb.weights")
    if isinstance(puzzle_embedding, torch.Tensor) and puzzle_embedding.ndim == 2:
        inferred["num_puzzle_identifiers"] = int(puzzle_embedding.shape[0])
        inferred["puzzle_emb_ndim"] = int(puzzle_embedding.shape[1])

    inferred.setdefault("puzzle_emb_ndim", int(inferred.get("hidden_size", DEFAULT_TRM_HIDDEN_SIZE)))

    max_layer_index = -1
    layer_prefix = "inner.L_level.layers."
    for key in state_dict:
        if not key.startswith(layer_prefix):
            continue
        suffix = key[len(layer_prefix) :]
        parts = suffix.split(".", 1)
        if not parts:
            continue
        try:
            layer_index = int(parts[0])
        except ValueError:
            continue
        max_layer_index = max(max_layer_index, layer_index)
    if max_layer_index >= 0:
        inferred["L_layers"] = max_layer_index + 1

    return inferred


def _select_compatible_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    model_state = model.state_dict()
    compatible_state_dict: dict[str, torch.Tensor] = {}
    unexpected_count = 0
    shape_mismatch_count = 0

    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            unexpected_count += 1
            continue
        if tuple(value.shape) != tuple(target.shape):
            shape_mismatch_count += 1
            continue
        compatible_state_dict[key] = value

    notes: list[str] = []
    if unexpected_count:
        notes.append(f"Skipped {unexpected_count} unexpected checkpoint tensors.")
    if shape_mismatch_count:
        notes.append(f"Skipped {shape_mismatch_count} shape-mismatched checkpoint tensors.")
    if compatible_state_dict:
        notes.append(f"Loaded {len(compatible_state_dict)} compatible checkpoint tensors.")
    return compatible_state_dict, notes


def _is_medical_compatible_config(config: SamsungTRMConfig) -> bool:
    return (
        config.seq_len == DEFAULT_TRM_SEQ_LEN
        and config.vocab_size == DEFAULT_TRM_VOCAB_SIZE
        and config.num_puzzle_identifiers >= DEFAULT_TRM_PUZZLE_IDENTIFIERS
        and config.hidden_size == DEFAULT_TRM_HIDDEN_SIZE
        and config.L_layers == DEFAULT_TRM_LAYERS
        and config.num_heads == DEFAULT_TRM_HEADS
        and config.forward_dtype.lower() in {"float16", "fp16"}
    )


def _resolve_simple_interpolations(payload: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(payload)
    for key, value in list(resolved.items()):
        if not isinstance(value, str):
            continue
        if not (value.startswith("${.") and value.endswith("}")):
            continue
        referenced_key = value[3:-1]
        if referenced_key in resolved:
            resolved[key] = resolved[referenced_key]
    return resolved


def _resolve_checkpoint_path(path: Path) -> Path | None:
    if path.is_file():
        return path
    if not path.exists():
        return None
    candidates = sorted(
        [candidate for pattern in ("*.pt", "*.pth", "*.ckpt", "*.bin") for candidate in path.rglob(pattern)],
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _resolve_preferred_checkpoint(repo_root: Path) -> Path | None:
    checkpoint_dir = repo_root / "checkpoints"
    if checkpoint_dir.exists():
        for preferred_name in PREFERRED_TRM_CHECKPOINT_NAMES:
            candidate = checkpoint_dir / preferred_name
            if candidate.exists():
                return candidate
        checkpoint = _resolve_checkpoint_path(checkpoint_dir)
        if checkpoint is not None:
            return checkpoint
    return _resolve_checkpoint_path(repo_root)


def _resolve_external_repo_root(path: Path) -> Path | None:
    if DEFAULT_EXTERNAL_TRM_REPO.exists():
        return DEFAULT_EXTERNAL_TRM_REPO
    if path.name == DEFAULT_EXTERNAL_TRM_REPO.name and path.exists():
        return path
    return None


def _resolve_repo_root(path: Path) -> Path | None:
    search_roots = []
    if path.exists():
        search_roots.append(path if path.is_dir() else path.parent)
    search_roots.extend(candidate for candidate in path.parents)
    for root in search_roots:
        if (root / OFFICIAL_TRM_SOURCE_FILE).exists():
            return root
        if (root / OFFICIAL_TRM_ALT_SOURCE_FILE).exists():
            return root
    return None


def _import_samsung_trm(repo_root: Path | None) -> type[nn.Module] | None:
    if repo_root is None:
        return None

    import_roots: list[Path] = [repo_root]
    if KAGGLE_WORKING_TRM_ROOT.exists() and KAGGLE_WORKING_TRM_ROOT not in import_roots:
        import_roots.append(KAGGLE_WORKING_TRM_ROOT)

    for import_root in import_roots:
        models_root = import_root / "models"
        module_candidates: list[str] = []
        try:
            module_path, _class_name = KagglePathResolver.get_trm_import_path(import_root)
            module_candidates.append(module_path)
        except FileNotFoundError:
            pass
        if (import_root / OFFICIAL_TRM_SOURCE_FILE).exists():
            for module_name in (OFFICIAL_TRM_MODULE, "recursive_reasoning.trm"):
                if module_name not in module_candidates:
                    module_candidates.append(module_name)
        if (import_root / OFFICIAL_TRM_ALT_SOURCE_FILE).exists() and "src.trm" not in module_candidates:
            module_candidates.append("src.trm")

        import_context = _temporary_sys_path(import_root)
        if models_root.exists():
            import_context = _nested_sys_paths(import_root, models_root)

        with import_context:
            try:
                for module_name in module_candidates:
                    try:
                        logger.info("Attempting TRM import from sys.path root: %s", import_root)
                        logger.info("Attempting TRM module import: %s", module_name)
                        module = importlib.import_module(module_name)
                    except Exception as exc:
                        logger.debug("TRM module import failed (%s from %s): %s", module_name, import_root, exc)
                        continue
                    model_class = _resolve_official_model_class(module)
                    if model_class is None:
                        logger.warning("No supported TRM class found in module %s", module_name)
                        continue
                    logger.info("Resolved TRM class %s from module %s", model_class.__name__, module_name)
                    return model_class

                source_path = import_root / OFFICIAL_TRM_SOURCE_FILE
                if not source_path.exists():
                    source_path = import_root / OFFICIAL_TRM_ALT_SOURCE_FILE
                if not source_path.exists():
                    continue
                dynamic_module_name = f"trm_dynamic_{abs(hash(str(source_path)))}"
                logger.info("Attempting dynamic TRM import from file: %s", source_path)
                spec = importlib.util.spec_from_file_location(dynamic_module_name, source_path)
                if spec is None or spec.loader is None:
                    logger.warning("Unable to create import spec for %s", source_path)
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[dynamic_module_name] = module
                spec.loader.exec_module(module)
                model_class = _resolve_official_model_class(module)
                if model_class is None:
                    logger.warning("Dynamic TRM import succeeded but no supported class found in %s", source_path)
                    continue
                logger.info("Resolved TRM class %s via dynamic import.", model_class.__name__)
                return model_class
            except Exception as exc:
                logger.warning("TRM import failed for root %s: %s", import_root, exc)
    return None


def _resolve_official_model_class(module: Any) -> type[nn.Module] | None:
    for class_name in OFFICIAL_TRM_CLASS_CANDIDATES:
        model_class = getattr(module, class_name, None)
        if model_class is not None:
            return model_class
    return None


@contextmanager
def _nested_sys_paths(primary: Path, secondary: Path) -> Iterator[None]:
    with _temporary_sys_path(primary):
        with _temporary_sys_path(secondary):
            yield


@contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    path_str = str(path)
    sys.path.insert(0, path_str)
    try:
        importlib.invalidate_caches()
        yield
    finally:
        if sys.path and sys.path[0] == path_str:
            sys.path.pop(0)


def _resolve_compute_dtype(forward_dtype: str, device: torch.device) -> torch.dtype:
    normalized = forward_dtype.lower()
    if device.type == "cuda" and normalized in {"float16", "fp16", "half", "bfloat16", "bf16"}:
        return torch.float16
    return torch.float32


__all__ = [
    "FallbackTinyRecursiveReasoningModel",
    "SamsungTRMConfig",
    "TRMCarry",
    "TRMInnerCarry",
    "TRMModelBundle",
    "adapt_samsung_checkpoint",
    "get_trm_paths",
    "initialize_official_medical_trm",
    "load_real_trm",
    "load_trm_model_bundle",
    "stablemax_cross_entropy",
    "verify_trm_repo",
]
