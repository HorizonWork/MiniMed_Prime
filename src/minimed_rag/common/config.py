"""Pseudocode for configuration loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class AppConfig:
    env: str
    graph_version: str
    values: dict[str, Any] = field(default_factory=dict)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        current: Any = self.values
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(env: str = "local", config_dir: str | Path = "configs") -> AppConfig:
    raw = load_yaml(Path(config_dir) / "env" / f"{env}.yaml")
    return AppConfig(
        env=str(raw.get("env", env)),
        graph_version=str(raw.get("graph_version", "kg_local")),
        values=raw,
    )
