from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.layers.vq_artifacts import (
    assert_metadata_compatibility,
    build_codebook_version,
    compute_schema_hash,
    compute_source_graph_signature,
    load_frozen_codebook,
    save_codebook_artifact,
    utc_now_iso,
)


def _artifact_metadata(tmp_path: Path) -> dict[str, object]:
    relation_hash = compute_schema_hash(("drug_drug", "indication"))
    node_type_hash = compute_schema_hash(("drug", "disease", "other"))
    source_signature = compute_source_graph_signature(tmp_path)
    created_at = utc_now_iso()
    version = build_codebook_version(
        created_at=created_at,
        codebook_size=8,
        hidden_dim=4,
        seed=42,
        relation_schema_hash=relation_hash,
        node_type_schema_hash=node_type_hash,
        source_graph_signature=source_signature,
    )
    return {
        "codebook_version": version,
        "created_at": created_at,
        "codebook_size": 8,
        "hidden_dim": 4,
        "num_subgraphs_seen": 10,
        "relation_schema_hash": relation_hash,
        "node_type_schema_hash": node_type_hash,
        "source_graph_signature": source_signature,
        "seed": 42,
    }


def test_save_and_load_codebook_artifact_roundtrip(tmp_path: Path) -> None:
    metadata = _artifact_metadata(tmp_path)
    codebook_weight = torch.arange(28, dtype=torch.float32).reshape(7, 4)
    save_codebook_artifact(
        tmp_path,
        codebook_weight=codebook_weight,
        metadata=metadata,
        stats={"dead_code_ratio": 0.0, "usage_counts_total": [1] * 7},
        progress={"status": "completed", "subgraphs_seen": 10},
        extra_state={"reconstruction_head_state_dict": {}},
    )

    loaded = load_frozen_codebook(tmp_path)

    assert loaded.metadata["codebook_version"] == metadata["codebook_version"]
    assert loaded.metadata["num_subgraphs_seen"] == 10
    assert torch.equal(loaded.codebook_weight, codebook_weight)


def test_metadata_compatibility_rejects_schema_mismatch(tmp_path: Path) -> None:
    metadata = _artifact_metadata(tmp_path)
    with pytest.raises(ValueError) as exc_info:
        assert_metadata_compatibility(
            metadata,
            expected_codebook_size=8,
            expected_hidden_dim=4,
            expected_relation_schema_hash="mismatch-relation-hash",
            expected_node_type_schema_hash=str(metadata["node_type_schema_hash"]),
            expected_source_graph_signature=str(metadata["source_graph_signature"]),
            strict_source_graph_signature=False,
        )
    assert "relation_schema_hash mismatch" in str(exc_info.value)
