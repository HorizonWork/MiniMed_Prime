from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.layers.layer3_trm import TRMReasoner
from src.models.trm_wrapper import FallbackTinyRecursiveReasoningModel, SamsungTRMConfig
from src.schemas import EvidenceBundle, TRMOutput


def make_reasoning_bundle() -> EvidenceBundle:
    return EvidenceBundle.model_validate(
        {
            "question_text": "Is warfarin indicated for atrial fibrillation, and does ibuprofen conflict with it?",
            "question_type": "drug_interaction",
            "question_entities": [
                {
                    "surface": "warfarin",
                    "cui": "C0043031",
                    "primekg_node_id": "drug:warfarin",
                    "entity_type": "drug",
                },
                {
                    "surface": "ibuprofen",
                    "cui": "C0020740",
                    "primekg_node_id": "drug:ibuprofen",
                    "entity_type": "drug",
                },
                {
                    "surface": "atrial fibrillation",
                    "cui": "C0004238",
                    "primekg_node_id": "disease:af",
                    "entity_type": "disease",
                },
            ],
            "subgraph_edges": [
                {
                    "edge_id": "E1",
                    "head": "drug:warfarin",
                    "tail": "disease:af",
                    "relation": "indication",
                    "display_relation": "is indicated for",
                    "source_reliability": 0.9,
                    "amg_confidence": 0.9,
                    "supporting_pmids": ["11111111"],
                },
                {
                    "edge_id": "E2",
                    "head": "drug:warfarin",
                    "tail": "drug:ibuprofen",
                    "relation": "drug_drug",
                    "display_relation": "interacts with",
                    "source_reliability": 0.9,
                    "amg_confidence": 0.3,
                    "supporting_pmids": ["22222222"],
                },
                {
                    "edge_id": "E3",
                    "head": "drug:ibuprofen",
                    "tail": "disease:bleeding",
                    "relation": "contraindication",
                    "display_relation": "is contraindicated in",
                    "source_reliability": 0.9,
                    "amg_confidence": 0.9,
                    "supporting_pmids": ["33333333"],
                },
            ],
            "pubmed_passages": [
                {
                    "pmid": "11111111",
                    "title": "Warfarin in atrial fibrillation",
                    "abstract": "Warfarin is indicated and effective in atrial fibrillation for stroke prevention.",
                    "relevance_score": 12.0,
                },
                {
                    "pmid": "99999999",
                    "title": "Conflicting warfarin evidence",
                    "abstract": "Warfarin is not effective and not recommended in atrial fibrillation in this cohort.",
                    "relevance_score": 9.5,
                },
            ],
            "metadata": {"question_type": "drug_interaction"},
        }
    )


def build_embedder_output(bundle: EvidenceBundle) -> dict[str, object]:
    inputs = torch.full((1, 256), 7, dtype=torch.long)
    inputs[0, :4] = torch.tensor([11, 12, 13, 14], dtype=torch.long)
    return {
        "inputs": inputs,
        "puzzle_identifiers": torch.tensor([0], dtype=torch.long),
        "codebook_indices": torch.tensor([11, 12, 13, 14] + [8191] * 252, dtype=torch.long),
        "node_mapping": {
            0: "drug:warfarin",
            1: "disease:af",
            2: "drug:ibuprofen",
            3: "disease:bleeding",
            **{index: "__pad__" for index in range(4, 256)},
        },
        "original_graph": bundle,
    }


def save_checkpoint(path: Path) -> None:
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


def test_trm_reasoner_loads_checkpoint_and_reasons(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "medical_trm.ckpt"
    save_checkpoint(checkpoint_path)
    bundle = make_reasoning_bundle()
    embedder_output = build_embedder_output(bundle)
    reasoner = TRMReasoner(model_path=str(checkpoint_path), device="cpu", max_steps=6, conf_threshold=0.45)

    output = reasoner.reason(embedder_output)

    assert isinstance(output, TRMOutput)
    assert reasoner.trm_is_real is False
    assert output.trace[0]["backend_used"] == "fallback_trm"
    assert output.trace[0]["trm_is_real"] is False
    assert 1 <= len(output.ranked_paths) <= 5
    assert len(output.trace) >= 1
    assert len(output.trace) <= 8


def test_trm_reasoner_returns_top_five_paths_when_available(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "medical_trm.ckpt"
    save_checkpoint(checkpoint_path)
    bundle = make_reasoning_bundle()
    embedder_output = build_embedder_output(bundle)
    reasoner = TRMReasoner(model_path=str(checkpoint_path), device="cpu", max_steps=5, conf_threshold=0.2)

    logits = torch.full((1, 256, 8192), -10.0)
    logits[0, 0, 11] = 5.0
    logits[0, 1, 12] = 5.0
    logits[0, 2, 13] = 5.0
    logits[0, 3, 14] = 5.0

    paths = reasoner.decode_paths(logits, embedder_output["node_mapping"], bundle.subgraph_edges, top_k=5)

    assert 1 <= len(paths) <= 5
    assert any(path.nodes[:2] == ["drug:warfarin", "disease:af"] for path in paths)


def test_multiplicative_confidence_penalizes_weak_edges() -> None:
    confidence = TRMReasoner.compute_multiplicative_confidence([0.9, 0.9, 0.3, 0.9])

    assert confidence == pytest.approx(0.177147, rel=1e-6)


def test_detect_contradiction_flags_conflicting_pubmed_evidence(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "medical_trm.ckpt"
    save_checkpoint(checkpoint_path)
    bundle = make_reasoning_bundle()
    reasoner = TRMReasoner(model_path=str(checkpoint_path), device="cpu")

    contradiction_score = reasoner.detect_contradiction(["drug:warfarin", "disease:af"], bundle)

    assert contradiction_score >= 0.5


def test_train_step_smoke(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "medical_trm.ckpt"
    save_checkpoint(checkpoint_path)
    reasoner = TRMReasoner(model_path=str(checkpoint_path), device="cpu", max_steps=3)
    labels = torch.full((1, 256), -100, dtype=torch.long)
    labels[0, :4] = torch.tensor([11, 12, 13, 14], dtype=torch.long)

    metrics = reasoner.train_step(
        {
            "trm_input": torch.full((1, 256), 7, dtype=torch.long),
            "gold_path_tokens": labels,
            "puzzle_ids": torch.tensor([0], dtype=torch.long),
        }
    )

    assert metrics["loss"] >= 0.0
    assert metrics["token_loss"] >= 0.0
    assert metrics["q_halt_loss"] >= 0.0
