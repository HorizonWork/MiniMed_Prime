from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .kaggle_env import KaggleEnv
from .structured_logger import DEFAULT_LOG_DIR, StructuredLogger


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())


class CheckpointManager:
    def __init__(self, save_dir: Path, keep_last_n: int = 3) -> None:
        self.save_dir = KaggleEnv.ensure_writeable(save_dir if save_dir.is_absolute() else KaggleEnv.path(save_dir))
        self.keep_last_n = keep_last_n
        self.logger = StructuredLogger("checkpoint_manager", DEFAULT_LOG_DIR)

    def save(
        self,
        step: int,
        model_state: dict[str, Any],
        optimizer_state: dict[str, Any],
        metrics: dict[str, Any],
        filepath: Path | None = None,
    ) -> Path:
        target_path = filepath or (self.save_dir / f"checkpoint_step_{step:08d}.pt")
        target_path = KaggleEnv.ensure_writeable(target_path)
        payload = {
            "step": int(step),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "metrics": dict(metrics),
        }
        torch.save(payload, target_path)
        self._prune()
        self.logger.log_checkpoint(target_path, step=step, metrics=metrics)
        if KaggleEnv.is_kaggle():
            self.logger.log_event(
                "kaggle_checkpoint_saved",
                {
                    "path": str(target_path),
                    "step": int(step),
                    "backend_used": "torch.save",
                    "latency_ms": None,
                    "reminder": "Download or persist /kaggle/working outputs after the run completes.",
                },
            )
        return target_path

    def latest(self) -> Path | None:
        if not self.save_dir.exists():
            return None
        checkpoints = sorted(self.save_dir.glob("*.pt"), key=lambda candidate: candidate.stat().st_mtime, reverse=True)
        return checkpoints[0] if checkpoints else None

    def load(self, checkpoint_path: Path) -> dict[str, Any]:
        payload = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"Checkpoint at {checkpoint_path} did not decode to a dictionary payload.")
        return payload

    def _prune(self) -> None:
        if self.keep_last_n <= 0 or not self.save_dir.exists():
            return
        checkpoints = sorted(self.save_dir.glob("*.pt"), key=lambda candidate: candidate.stat().st_mtime, reverse=True)
        for stale_checkpoint in checkpoints[self.keep_last_n :]:
            try:
                stale_checkpoint.unlink()
            except FileNotFoundError:
                continue


def resume_or_initialize(model: Any, optimizer: Any, checkpoint_dir: Path) -> int:
    manager = CheckpointManager(checkpoint_dir)
    latest_checkpoint = manager.latest()
    if latest_checkpoint is None:
        return 0

    payload = manager.load(latest_checkpoint)
    model_state = payload.get("model_state", {})
    optimizer_state = payload.get("optimizer_state", {})
    if hasattr(model, "load_state_dict"):
        model.load_state_dict(model_state)
    if optimizer is not None and hasattr(optimizer, "load_state_dict"):
        optimizer.load_state_dict(optimizer_state)
    step = int(payload.get("step", -1))
    manager.logger.log_event(
        "checkpoint_resumed",
        {
            "path": str(latest_checkpoint),
            "step": step,
            "backend_used": "torch.load",
            "latency_ms": None,
        },
    )
    return step + 1


__all__ = ["CheckpointManager", "resume_or_initialize"]
