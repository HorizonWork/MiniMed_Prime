from __future__ import annotations

import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from .kaggle_env import KaggleEnv


DEFAULT_LOG_DIR = Path("data/logs")
DEFAULT_SESSION_ID = uuid4().hex


class StructuredLogger:
    def __init__(self, name: str, log_dir: Path) -> None:
        self.name = name
        self.session_id = DEFAULT_SESSION_ID
        resolved_dir = KaggleEnv.ensure_writeable(log_dir if log_dir.is_absolute() else KaggleEnv.path(log_dir))
        self.log_dir = resolved_dir
        self.log_path = self.log_dir / f"{self._sanitize_filename(name)}.jsonl"
        self._lock = Lock()
        self._console_logger = logging.getLogger(f"minimed.{name}")
        if not self._console_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s %(message)s"))
            self._console_logger.addHandler(handler)
            self._console_logger.setLevel(logging.INFO)
            self._console_logger.propagate = False

    def log_event(self, event: str, data: dict[str, Any]) -> None:
        payload_data = dict(data)
        payload_data.setdefault("backend_used", None)
        payload_data.setdefault("latency_ms", None)
        payload = {
            "timestamp": self._timestamp(),
            "layer": self.name,
            "event": event,
            "data": self._json_safe(payload_data),
            "session_id": self.session_id,
        }
        self._write_payload(payload)
        self._console_logger.info("%s %s", event, json.dumps(payload["data"], sort_keys=True))

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        self.log_event(
            "metric",
            {
                "name": name,
                "value": float(value),
                "step": step,
                "latency_ms": None,
            },
        )

    def log_checkpoint(self, path: Path, step: int, metrics: dict[str, Any]) -> None:
        self.log_event(
            "checkpoint",
            {
                "path": str(path),
                "step": int(step),
                "metrics": dict(metrics),
                "latency_ms": None,
            },
        )

    def log_exception(self, event: str, exc: BaseException, data: dict[str, Any] | None = None) -> None:
        payload_data = dict(data or {})
        payload_data.update(
            {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        self.log_event(event, payload_data)

    def _write_payload(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        cleaned = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in name)
        return cleaned.strip("_") or "layer"

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._json_safe(item) for item in value]
        if hasattr(value, "shape") and hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                return str(value)
        return str(value)


__all__ = ["DEFAULT_LOG_DIR", "StructuredLogger"]
