from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.models.trm_wrapper import DEFAULT_IGNORE_LABEL_ID, FallbackTinyRecursiveReasoningModel, SamsungTRMConfig
from src.training import TRMArrayDataset, TRMTrainerConfig, train_medical_trm


def _write_split(root: Path, split: str) -> None:
    split_dir = root / split
    split_dir.mkdir(parents=True)
    inputs = np.full((1, 256), 7, dtype=np.int64)
    inputs[0, :3] = np.asarray([11, 12, 13], dtype=np.int64)
    labels = np.full((1, 256), DEFAULT_IGNORE_LABEL_ID, dtype=np.int64)
    labels[0, :3] = np.asarray([11, 12, 13], dtype=np.int64)
    puzzle_identifiers = np.asarray([0], dtype=np.int64)
    np.save(split_dir / "all__inputs.npy", inputs, allow_pickle=False)
    np.save(split_dir / "all__labels.npy", labels, allow_pickle=False)
    np.save(split_dir / "all__puzzle_identifiers.npy", puzzle_identifiers, allow_pickle=False)


def _save_medical_checkpoint(path: Path) -> None:
    config = SamsungTRMConfig(
        batch_size=1,
        seq_len=256,
        num_puzzle_identifiers=5,
        vocab_size=8192,
        H_cycles=3,
        L_cycles=4,
        L_layers=2,
        hidden_size=512,
        num_heads=8,
        forward_dtype="float16",
    )
    model = FallbackTinyRecursiveReasoningModel(config)
    torch.save({"config": config.to_model_dict(), "state_dict": model.state_dict()}, path)


def test_trm_array_dataset_loads_prepared_split(tmp_path: Path) -> None:
    dataset_root = tmp_path / "trm_medical"
    _write_split(dataset_root, "train")

    dataset = TRMArrayDataset(dataset_root / "train")

    assert len(dataset) == 1
    item = dataset[0]
    assert item["trm_input"].shape == (256,)
    assert item["gold_path_tokens"][:3].tolist() == [11, 12, 13]
    assert item["puzzle_ids"].item() == 0


def test_train_medical_trm_smoke_saves_final_checkpoint(tmp_path: Path) -> None:
    dataset_root = tmp_path / "trm_medical"
    _write_split(dataset_root, "train")
    _write_split(dataset_root, "val")
    checkpoint_path = tmp_path / "medical_seed.ckpt"
    _save_medical_checkpoint(checkpoint_path)

    result = train_medical_trm(
        TRMTrainerConfig(
            dataset_dir=dataset_root,
            model_path=str(checkpoint_path),
            output_dir=tmp_path / "out",
            device="cpu",
            epochs=1,
            batch_size=1,
            max_steps=1,
            save_every=9999,
            eval_every=1,
            max_train_batches=1,
            max_val_batches=1,
            allow_official_untrained=False,
        )
    )

    assert result.status == "SUCCESS"
    assert result.steps_completed == 1
    assert result.final_checkpoint.exists()
    assert result.train_metrics["loss"] >= 0.0
    assert result.validation_metrics["examples"] == 1.0
