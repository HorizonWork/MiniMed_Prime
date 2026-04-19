from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.layers.layer3_trm import DEFAULT_HALT_LOSS_WEIGHT, TRMReasoner
from src.models.trm_wrapper import DEFAULT_IGNORE_LABEL_ID, stablemax_cross_entropy
from src.utils.kaggle_env import KaggleEnv
from src.utils.structured_logger import DEFAULT_LOG_DIR, StructuredLogger


DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_VAL_SPLIT = "val"
DEFAULT_FINAL_CHECKPOINT_NAME = "medical_trm.pt"


@dataclass(slots=True)
class TRMTrainerConfig:
    dataset_dir: Path
    model_path: str = "external/TinyRecursiveModels"
    output_dir: Path = Path("data/checkpoints/active")
    train_split: str = DEFAULT_TRAIN_SPLIT
    val_split: str | None = DEFAULT_VAL_SPLIT
    final_checkpoint_name: str = DEFAULT_FINAL_CHECKPOINT_NAME
    device: str = "cpu"
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1.0
    max_steps: int = 8
    save_every: int = 2500
    eval_every: int = 5000
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    allow_official_untrained: bool = True
    seed: int = 13


@dataclass(slots=True)
class TRMTrainingRunResult:
    status: str
    final_checkpoint: Path
    steps_completed: int
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    dataset_stats: dict[str, Any]
    backend_used: str
    trm_is_real: bool

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["final_checkpoint"] = str(self.final_checkpoint)
        return payload


class TRMArrayDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, split_dir: Path) -> None:
        self.split_dir = split_dir
        self.inputs = _load_array(split_dir / "all__inputs.npy", dtype=np.int64)
        self.labels = _load_array(split_dir / "all__labels.npy", dtype=np.int64)
        self.puzzle_identifiers = _load_array(split_dir / "all__puzzle_identifiers.npy", dtype=np.int64)
        if self.inputs.ndim != 2:
            raise ValueError(f"Expected inputs with shape [N,L], received {self.inputs.shape}.")
        if self.labels.shape != self.inputs.shape:
            raise ValueError(f"Expected labels shape {self.inputs.shape}, received {self.labels.shape}.")
        if self.puzzle_identifiers.shape != (self.inputs.shape[0],):
            raise ValueError(
                f"Expected puzzle identifiers shape ({self.inputs.shape[0]},), received {self.puzzle_identifiers.shape}."
            )

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "trm_input": torch.as_tensor(self.inputs[index], dtype=torch.long),
            "gold_path_tokens": torch.as_tensor(self.labels[index], dtype=torch.long),
            "puzzle_ids": torch.as_tensor(self.puzzle_identifiers[index], dtype=torch.long),
        }


def train_medical_trm(config: TRMTrainerConfig) -> TRMTrainingRunResult:
    started_at = time.perf_counter()
    logger = StructuredLogger("medical_trm_trainer", DEFAULT_LOG_DIR)
    torch.manual_seed(config.seed)

    dataset_root = _resolve_existing_path(config.dataset_dir)
    train_split_dir = resolve_trm_split_dir(dataset_root, config.train_split)
    val_split_dir = resolve_trm_split_dir(dataset_root, config.val_split) if config.val_split else None
    train_dataset = TRMArrayDataset(train_split_dir)
    val_dataset = TRMArrayDataset(val_split_dir) if val_split_dir is not None and val_split_dir.exists() else None
    dataset_stats = _dataset_stats(train_dataset=train_dataset, val_dataset=val_dataset)

    output_dir = KaggleEnv.ensure_writeable(_resolve_output_path(config.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = KaggleEnv.ensure_writeable(output_dir / "resume")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    reasoner = TRMReasoner(
        model_path=config.model_path,
        device=config.device,
        max_steps=config.max_steps,
        allow_official_untrained=config.allow_official_untrained,
    )
    reasoner.optimizer = torch.optim.AdamW(
        reasoner.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    logger.log_event(
        "training_start",
        {
            "config": _config_for_log(config),
            "dataset_stats": dataset_stats,
            "model_parameter_count": _parameter_count(reasoner.model),
            "model_source": reasoner.model_bundle.source,
            "checkpoint_loaded": reasoner.model_bundle.checkpoint_loaded,
            "medical_compatible": reasoner.model_bundle.medical_compatible,
            "trm_is_real": reasoner.trm_is_real,
            "backend_used": reasoner.backend_used,
            "latency_ms": 0.0,
        },
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )

    global_step = 0
    last_train_metrics: dict[str, float] = {}
    last_validation_metrics: dict[str, float] = {}
    final_checkpoint = output_dir / config.final_checkpoint_name

    for epoch_index in range(config.epochs):
        epoch_started_at = time.perf_counter()
        logger.log_event(
            "epoch_start",
            {
                "epoch": epoch_index + 1,
                "epochs": config.epochs,
                "batches": len(train_loader),
                "backend_used": reasoner.backend_used,
                "latency_ms": 0.0,
            },
        )

        for batch_index, batch in enumerate(train_loader):
            if config.max_train_batches is not None and batch_index >= config.max_train_batches:
                break
            global_step += 1
            last_train_metrics = reasoner.train_step(
                batch,
                step=global_step,
                checkpoint_dir=checkpoint_dir,
                save_every=config.save_every,
            )
            if config.eval_every > 0 and val_loader is not None and global_step % config.eval_every == 0:
                last_validation_metrics = evaluate_reasoner(
                    reasoner=reasoner,
                    loader=val_loader,
                    max_batches=config.max_val_batches,
                )
                _log_validation(logger, reasoner, global_step, last_validation_metrics)
                save_medical_trm_checkpoint(
                    reasoner=reasoner,
                    output_path=output_dir / f"medical_trm_step_{global_step:08d}.pt",
                    metrics={**last_train_metrics, **{f"val_{key}": value for key, value in last_validation_metrics.items()}},
                    step=global_step,
                )

            if config.save_every > 0 and global_step % config.save_every == 0:
                save_medical_trm_checkpoint(
                    reasoner=reasoner,
                    output_path=output_dir / f"medical_trm_step_{global_step:08d}.pt",
                    metrics=last_train_metrics,
                    step=global_step,
                )

        logger.log_event(
            "epoch_complete",
            {
                "epoch": epoch_index + 1,
                "steps_completed": global_step,
                "train_metrics": last_train_metrics,
                "backend_used": reasoner.backend_used,
                "latency_ms": (time.perf_counter() - epoch_started_at) * 1000.0,
            },
        )

    if val_loader is not None:
        last_validation_metrics = evaluate_reasoner(
            reasoner=reasoner,
            loader=val_loader,
            max_batches=config.max_val_batches,
        )
        _log_validation(logger, reasoner, global_step, last_validation_metrics)

    save_medical_trm_checkpoint(
        reasoner=reasoner,
        output_path=final_checkpoint,
        metrics={**last_train_metrics, **{f"val_{key}": value for key, value in last_validation_metrics.items()}},
        step=global_step,
    )

    result = TRMTrainingRunResult(
        status="SUCCESS" if global_step > 0 else "NO_STEPS",
        final_checkpoint=final_checkpoint,
        steps_completed=global_step,
        train_metrics=last_train_metrics,
        validation_metrics=last_validation_metrics,
        dataset_stats=dataset_stats,
        backend_used=reasoner.backend_used,
        trm_is_real=reasoner.trm_is_real,
    )
    logger.log_event(
        "training_complete",
        {
            **result.to_json_dict(),
            "backend_used": reasoner.backend_used,
            "latency_ms": (time.perf_counter() - started_at) * 1000.0,
        },
    )
    return result


def evaluate_reasoner(
    *,
    reasoner: TRMReasoner,
    loader: DataLoader[dict[str, torch.Tensor]],
    max_batches: int | None = None,
) -> dict[str, float]:
    was_training = reasoner.model.training
    reasoner.model.eval()
    total_loss = 0.0
    total_token_loss = 0.0
    total_halt_loss = 0.0
    total_valid_tokens = 0
    total_correct_tokens = 0
    total_sequences = 0
    total_correct_sequences = 0
    batches = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            metrics = _evaluate_batch(reasoner=reasoner, batch=batch)
            batches += 1
            batch_size = int(metrics["batch_size"])
            total_loss += metrics["loss"] * batch_size
            total_token_loss += metrics["token_loss"] * batch_size
            total_halt_loss += metrics["q_halt_loss"] * batch_size
            total_valid_tokens += int(metrics["valid_token_count"])
            total_correct_tokens += int(metrics["correct_token_count"])
            total_sequences += batch_size
            total_correct_sequences += int(metrics["correct_sequence_count"])

    if was_training:
        reasoner.model.train()
    else:
        reasoner.model.eval()

    denominator = max(total_sequences, 1)
    return {
        "loss": total_loss / denominator,
        "token_loss": total_token_loss / denominator,
        "q_halt_loss": total_halt_loss / denominator,
        "token_accuracy": total_correct_tokens / max(total_valid_tokens, 1),
        "sequence_accuracy": total_correct_sequences / denominator,
        "valid_token_count": float(total_valid_tokens),
        "batches": float(batches),
        "examples": float(total_sequences),
    }


def save_medical_trm_checkpoint(
    *,
    reasoner: TRMReasoner,
    output_path: Path,
    metrics: dict[str, float],
    step: int,
) -> Path:
    target_path = KaggleEnv.ensure_writeable(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": reasoner.model_bundle.config.to_model_dict(),
        "state_dict": reasoner.model.state_dict(),
        "metrics": dict(metrics),
        "step": int(step),
        "backend_used": reasoner.backend_used,
        "trm_is_real": reasoner.trm_is_real,
        "medical_compatible": reasoner.model_bundle.medical_compatible,
        "checkpoint_loaded": reasoner.model_bundle.checkpoint_loaded,
    }
    torch.save(payload, target_path)
    StructuredLogger("medical_trm_trainer", DEFAULT_LOG_DIR).log_checkpoint(target_path, step=step, metrics=metrics)
    return target_path


def resolve_trm_split_dir(dataset_dir: Path, split: str | None) -> Path:
    root = _resolve_existing_path(dataset_dir)
    if split is None:
        return root
    split_candidate = root / split
    if split_candidate.exists():
        return split_candidate
    if (root / "all__inputs.npy").exists():
        return root
    raise FileNotFoundError(f"TRM dataset split '{split}' was not found under {root}.")


def _evaluate_batch(reasoner: TRMReasoner, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    inputs = batch["trm_input"].to(reasoner.device)
    labels = batch["gold_path_tokens"].to(reasoner.device)
    puzzle_identifiers = batch["puzzle_ids"].to(reasoner.device)
    train_batch = {
        "inputs": inputs,
        "labels": labels,
        "puzzle_identifiers": puzzle_identifiers,
    }
    carry = reasoner.model.initial_carry(train_batch)
    final_outputs: dict[str, torch.Tensor] | None = None
    for _ in range(reasoner.max_steps):
        carry, outputs = reasoner.model(carry, train_batch)
        final_outputs = outputs
        if carry.halted.all():
            break
    if final_outputs is None:
        raise RuntimeError("TRM validation produced no outputs.")

    logits = final_outputs["logits"]
    token_loss = stablemax_cross_entropy(logits, labels)
    predictions = logits.argmax(dim=-1)
    valid_mask = labels != DEFAULT_IGNORE_LABEL_ID
    if torch.any(valid_mask):
        seq_is_correct = ((predictions == labels) | ~valid_mask).all(dim=-1)
        correct_token_count = int((predictions[valid_mask] == labels[valid_mask]).sum().detach().cpu().item())
    else:
        seq_is_correct = torch.ones((labels.shape[0],), dtype=torch.bool, device=labels.device)
        correct_token_count = int(labels.shape[0])
    q_halt_loss = F.binary_cross_entropy_with_logits(
        final_outputs["q_halt_logits"],
        seq_is_correct.to(dtype=final_outputs["q_halt_logits"].dtype),
    )
    loss = token_loss + (DEFAULT_HALT_LOSS_WEIGHT * q_halt_loss)
    return {
        "loss": float(loss.detach().cpu()),
        "token_loss": float(token_loss.detach().cpu()),
        "q_halt_loss": float(q_halt_loss.detach().cpu()),
        "batch_size": float(inputs.shape[0]),
        "valid_token_count": float(valid_mask.sum().detach().cpu().item()),
        "correct_token_count": float(correct_token_count),
        "correct_sequence_count": float(seq_is_correct.sum().detach().cpu().item()),
    }


def _load_array(path: Path, *, dtype: Any) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing TRM dataset array: {path}")
    return np.load(path, allow_pickle=False).astype(dtype, copy=False)


def _resolve_existing_path(path: Path) -> Path:
    resolved = path if path.is_absolute() else KaggleEnv.path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved


def _resolve_output_path(path: Path) -> Path:
    return path if path.is_absolute() else KaggleEnv.path(path)


def _dataset_stats(*, train_dataset: TRMArrayDataset, val_dataset: TRMArrayDataset | None) -> dict[str, Any]:
    stats = {
        "train_examples": len(train_dataset),
        "train_seq_len": int(train_dataset.inputs.shape[1]),
        "train_valid_label_tokens": int((train_dataset.labels != DEFAULT_IGNORE_LABEL_ID).sum()),
        "train_mean_valid_label_tokens": float((train_dataset.labels != DEFAULT_IGNORE_LABEL_ID).sum(axis=1).mean())
        if len(train_dataset)
        else 0.0,
        "train_unique_puzzle_ids": sorted(int(item) for item in np.unique(train_dataset.puzzle_identifiers).tolist()),
    }
    if val_dataset is not None:
        stats.update(
            {
                "val_examples": len(val_dataset),
                "val_seq_len": int(val_dataset.inputs.shape[1]),
                "val_valid_label_tokens": int((val_dataset.labels != DEFAULT_IGNORE_LABEL_ID).sum()),
                "val_mean_valid_label_tokens": float((val_dataset.labels != DEFAULT_IGNORE_LABEL_ID).sum(axis=1).mean())
                if len(val_dataset)
                else 0.0,
                "val_unique_puzzle_ids": sorted(int(item) for item in np.unique(val_dataset.puzzle_identifiers).tolist()),
            }
        )
    return stats


def _log_validation(
    logger: StructuredLogger,
    reasoner: TRMReasoner,
    step: int,
    metrics: dict[str, float],
) -> None:
    for metric_name, metric_value in metrics.items():
        if metric_name in {"loss", "token_accuracy", "sequence_accuracy"}:
            logger.log_metric(f"val_{metric_name}", metric_value, step=step)
    logger.log_event(
        "validation_complete",
        {
            "step": step,
            "metrics": metrics,
            "backend_used": reasoner.backend_used,
            "latency_ms": 0.0,
        },
    )


def _parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _config_for_log(config: TRMTrainerConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("dataset_dir", "output_dir"):
        payload[key] = str(payload[key])
    return payload


__all__ = [
    "TRMArrayDataset",
    "TRMTrainerConfig",
    "TRMTrainingRunResult",
    "evaluate_reasoner",
    "resolve_trm_split_dir",
    "save_medical_trm_checkpoint",
    "train_medical_trm",
]
