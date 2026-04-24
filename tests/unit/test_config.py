"""Smoke tests for config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from minimed_rag.common.config import AppConfig, load_config, load_yaml


def test_load_yaml_local(configs_dir: Path):
    data = load_yaml(configs_dir / "env" / "local.yaml")
    assert isinstance(data, dict)
    assert data["env"] == "local"


def test_load_config_local(configs_dir: Path):
    cfg = load_config(env="local", config_dir=configs_dir)
    assert isinstance(cfg, AppConfig)
    assert cfg.env == "local"
    assert cfg.graph_version == "kg_local"


def test_load_config_get_missing_key(configs_dir: Path):
    cfg = load_config(env="local", config_dir=configs_dir)
    assert cfg.get("nonexistent.key", "fallback") == "fallback"


def test_load_config_get_existing_key(configs_dir: Path):
    cfg = load_config(env="local", config_dir=configs_dir)
    assert cfg.get("env") == "local"


@pytest.mark.parametrize("env_name", ["local", "dev", "staging", "prod"])
def test_all_env_configs_parse(env_name: str, configs_dir: Path):
    cfg = load_config(env=env_name, config_dir=configs_dir)
    assert isinstance(cfg, AppConfig)
    assert cfg.env
