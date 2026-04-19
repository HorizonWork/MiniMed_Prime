from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.utils import CheckpointManager, KaggleEnv, StructuredLogger, T4Hardening, resume_or_initialize


def test_kaggle_env_path_returns_local_path_outside_kaggle() -> None:
    resolved = KaggleEnv.path("data/logs")

    assert resolved == Path("data/logs")


def test_kaggle_env_aliases_old_llm_path_to_medreason_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    medreason_root = input_root / "medreason-8b"
    medreason_root.mkdir(parents=True)

    monkeypatch.setattr(KaggleEnv, "INPUT_ROOT", input_root)
    monkeypatch.setattr(KaggleEnv, "is_kaggle", staticmethod(lambda: True))

    resolved = KaggleEnv.path("data/checkpoints/llm")

    assert resolved == medreason_root


def test_kaggle_env_aliases_medreason_dataset_on_kaggle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    medreason_root = input_root / "medreason"
    medreason_root.mkdir(parents=True)

    monkeypatch.setattr(KaggleEnv, "INPUT_ROOT", input_root)
    monkeypatch.setattr(KaggleEnv, "is_kaggle", staticmethod(lambda: True))

    resolved = KaggleEnv.path("data/medreason")

    assert resolved == medreason_root


def test_kaggle_env_resolves_primekg_via_dataset_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    primekg_root = input_root / "primekg-medical-knowledge-graph"
    primekg_root.mkdir(parents=True)
    (primekg_root / "dataset-metadata.json").write_text(
        json.dumps({"id": "huynhnhuthuyk18hcm/primekg", "title": "PrimeKG Medical Knowledge Graph"}),
        encoding="utf-8",
    )
    (primekg_root / "edges.csv").write_text("x_index,y_index,relation,display_relation\n", encoding="utf-8")
    (primekg_root / "nodes.csv").write_text("node_index,node_id,node_name,node_type\n", encoding="utf-8")

    monkeypatch.setattr(KaggleEnv, "INPUT_ROOT", input_root)
    monkeypatch.setattr(KaggleEnv, "is_kaggle", staticmethod(lambda: True))

    resolved = KaggleEnv.path("data/kg/primekg")

    assert resolved == primekg_root


def test_kaggle_env_resolves_primekg_via_signature_without_matching_slug(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    primekg_root = input_root / "mounted-dataset-123"
    primekg_root.mkdir(parents=True)
    (primekg_root / "edges.csv").write_text("x_index,y_index,relation,display_relation\n", encoding="utf-8")
    (primekg_root / "nodes.csv").write_text("node_index,node_id,node_name,node_type\n", encoding="utf-8")

    monkeypatch.setattr(KaggleEnv, "INPUT_ROOT", input_root)
    monkeypatch.setattr(KaggleEnv, "is_kaggle", staticmethod(lambda: True))

    resolved = KaggleEnv.path("data/kg/primekg")

    assert resolved == primekg_root


def test_kaggle_env_aliases_medreason_dataset_to_local_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    asset_root = tmp_path / "assets"
    medreason_root = asset_root / "medreason-upload"
    medreason_root.mkdir(parents=True)

    monkeypatch.setattr(KaggleEnv, "LOCAL_ASSET_ROOT", asset_root)
    monkeypatch.setattr(KaggleEnv, "is_kaggle", staticmethod(lambda: False))

    resolved = KaggleEnv.path("data/medreason")

    assert resolved == medreason_root


def test_kaggle_env_resolves_nested_trm_real_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    working_root = tmp_path / "working"
    (working_root / "external" / "TinyRecursiveModels").mkdir(parents=True)
    trm_repo = input_root / "trm-real" / "TinyRecursiveModels_bundle" / "TinyRecursiveModels"
    trm_module = trm_repo / "models" / "recursive_reasoning" / "trm.py"
    trm_module.parent.mkdir(parents=True)
    trm_module.write_text("# test trm module\n", encoding="utf-8")
    checkpoint_dir = trm_repo / "checkpoints"
    checkpoint_dir.mkdir()

    monkeypatch.setattr(KaggleEnv, "INPUT_ROOT", input_root)
    monkeypatch.setattr(KaggleEnv, "WORKING_ROOT", working_root)
    monkeypatch.setattr(KaggleEnv, "is_kaggle", staticmethod(lambda: True))

    assert KaggleEnv.path("external/TinyRecursiveModels") == trm_repo
    assert KaggleEnv.path("external/TinyRecursiveModels/checkpoints") == checkpoint_dir


def test_kaggle_env_matches_encoder_style_mount_names(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    article_root = input_root / "medcpt-article-encoder"
    article_root.mkdir(parents=True)
    (article_root / "config.json").write_text("{}", encoding="utf-8")
    (article_root / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(KaggleEnv, "INPUT_ROOT", input_root)
    monkeypatch.setattr(KaggleEnv, "is_kaggle", staticmethod(lambda: True))

    resolved = KaggleEnv.path("data/checkpoints/medcpt-article")

    assert resolved == article_root


def test_kaggle_env_resolves_trm_dataset_with_descriptive_mount_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    working_root = tmp_path / "working"
    (working_root / "external" / "TinyRecursiveModels").mkdir(parents=True)
    dataset_root = input_root / "tinyrecursivemodels-real-repo-and-arc"
    trm_repo = dataset_root / "TinyRecursiveModels_bundle" / "TinyRecursiveModels"
    trm_module = trm_repo / "models" / "recursive_reasoning" / "trm.py"
    trm_module.parent.mkdir(parents=True)
    trm_module.write_text("# test trm module\n", encoding="utf-8")

    monkeypatch.setattr(KaggleEnv, "INPUT_ROOT", input_root)
    monkeypatch.setattr(KaggleEnv, "WORKING_ROOT", working_root)
    monkeypatch.setattr(KaggleEnv, "is_kaggle", staticmethod(lambda: True))

    resolved = KaggleEnv.path("external/TinyRecursiveModels")

    assert resolved == trm_repo


def test_kaggle_env_iterates_nested_datasets_mount_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    nested_dataset = input_root / "datasets" / "huynhnhuthuyk18hcm" / "medcpt-article"
    nested_dataset.mkdir(parents=True)
    (nested_dataset / "config.json").write_text("{}", encoding="utf-8")
    (nested_dataset / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(KaggleEnv, "INPUT_ROOT", input_root)
    monkeypatch.setattr(KaggleEnv, "is_kaggle", staticmethod(lambda: True))

    resolved = KaggleEnv.path("data/checkpoints/medcpt-article")

    assert resolved == nested_dataset


def test_kaggle_env_prefers_nested_dataset_version_directory_when_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    versioned_dataset = input_root / "datasets" / "huynhnhuthuyk18hcm" / "medreason-8b" / "versions" / "3"
    versioned_dataset.mkdir(parents=True)
    (versioned_dataset / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(KaggleEnv, "INPUT_ROOT", input_root)
    monkeypatch.setattr(KaggleEnv, "is_kaggle", staticmethod(lambda: True))

    resolved = KaggleEnv.path("data/checkpoints/medreason-8b")

    assert resolved == versioned_dataset


def test_structured_logger_writes_jsonl_payload(tmp_path: Path) -> None:
    logger = StructuredLogger("test_layer", tmp_path)

    logger.log_event("demo_event", {"backend_used": "unit_test", "latency_ms": 1.5, "value": 3})

    log_path = tmp_path / "test_layer.jsonl"
    assert log_path.exists()
    payload = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert payload["layer"] == "test_layer"
    assert payload["event"] == "demo_event"
    assert payload["data"]["backend_used"] == "unit_test"
    assert payload["data"]["latency_ms"] == 1.5


def test_checkpoint_manager_save_load_and_resume(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    manager = CheckpointManager(tmp_path, keep_last_n=2)

    saved_path = manager.save(
        step=4,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        metrics={"loss": 0.25},
    )

    assert saved_path.exists()
    payload = manager.load(saved_path)
    assert payload["step"] == 4
    assert payload["metrics"]["loss"] == 0.25
    assert manager.latest() == saved_path

    resumed_model = torch.nn.Linear(4, 2)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-3)
    next_step = resume_or_initialize(resumed_model, resumed_optimizer, tmp_path)

    assert next_step == 5


def test_t4_hardening_rejects_bf16_requests() -> None:
    with pytest.raises(ValueError):
        T4Hardening.verify_fp16("bf16")
