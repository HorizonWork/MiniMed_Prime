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


@pytest.mark.parametrize(
    "config_name",
    ["bm25_only.yaml", "dense_only.yaml", "hybrid.yaml", "hybrid_rerank.yaml"],
)
def test_retrieval_ablation_configs_parse(config_name: str, configs_dir: Path):
    data = load_yaml(configs_dir / "retrieval" / config_name)
    assert data["retrieval"]["top_k"] == 10
    assert data["retrieval"]["retriever"]


def test_retrieval_ablation_config_maps_to_cli_overrides(configs_dir: Path):
    from minimed_rag.cli.evaluate import _load_retrieval_overrides

    overrides = _load_retrieval_overrides(str(configs_dir / "retrieval" / "hybrid_rerank.yaml"))

    assert overrides["retriever"] == "hybrid_rerank"
    assert overrides["index_path"] == "artifacts/indexes/bm25_textbooks.pkl"
    assert overrides["dense_index_path"] == "artifacts/indexes/dense_textbooks_bge-small-en-v1.5"
    assert overrides["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert overrides["reranker_model"] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
