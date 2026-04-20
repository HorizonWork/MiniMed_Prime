from __future__ import annotations

import torch

from src.layers.layer2_embedder import DEFAULT_PAD_IDENTIFIER, MedicalGraphEmbedder
from src.layers.vq_artifacts import estimate_token_stability, inspect_codebook_usage, save_codebook_artifact
from src.schemas import EvidenceBundle


def make_evidence_bundle() -> EvidenceBundle:
    return EvidenceBundle.model_validate(
        {
            "question_text": "Is ibuprofen contraindicated with warfarin in atrial fibrillation?",
            "question_type": "drug_interaction",
            "question_entities": [
                {
                    "surface": "ibuprofen",
                    "cui": "C0020740",
                    "primekg_node_id": "drug:ibuprofen",
                    "entity_type": "drug",
                },
                {
                    "surface": "warfarin",
                    "cui": "C0043031",
                    "primekg_node_id": "drug:warfarin",
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
                    "tail": "drug:ibuprofen",
                    "relation": "drug_drug",
                    "display_relation": "drug interaction",
                    "source_reliability": 0.95,
                    "amg_confidence": 0.9,
                    "supporting_pmids": ["12345678"],
                },
                {
                    "edge_id": "E2",
                    "head": "drug:warfarin",
                    "tail": "disease:af",
                    "relation": "indication",
                    "display_relation": "indication",
                    "source_reliability": 0.9,
                    "amg_confidence": 0.85,
                    "supporting_pmids": ["23456789"],
                },
                {
                    "edge_id": "E3",
                    "head": "disease:af",
                    "tail": "disease:stroke",
                    "relation": "disease_disease",
                    "display_relation": "associated disease",
                    "source_reliability": 0.78,
                    "amg_confidence": 0.8,
                    "supporting_pmids": ["34567890"],
                },
            ],
            "pubmed_passages": [
                {
                    "pmid": "12345678",
                    "title": "Warfarin and ibuprofen",
                    "abstract": "Ibuprofen increases bleeding risk when combined with warfarin.",
                    "relevance_score": 12.0,
                },
                {
                    "pmid": "23456789",
                    "title": "Warfarin in atrial fibrillation",
                    "abstract": "Warfarin reduces embolic stroke risk in atrial fibrillation.",
                    "relevance_score": 10.0,
                },
            ],
            "metadata": {"question_type": "drug_interaction"},
        }
    )


def build_embedder() -> MedicalGraphEmbedder:
    return MedicalGraphEmbedder(
        hidden_dim=64,
        codebook_size=128,
        max_len=256,
        sapbert_model_name=None,
        medcpt_article_model_name=None,
    )


def test_medical_graph_embedder_returns_trm_compatible_shapes() -> None:
    bundle = make_evidence_bundle()
    embedder = build_embedder()

    outputs = embedder(bundle)

    assert outputs["inputs"].shape == (1, 256)
    assert outputs["puzzle_identifiers"].shape == (1,)
    assert outputs["codebook_indices"].shape == (256,)
    assert outputs["inputs"].dtype == torch.long
    assert outputs["puzzle_identifiers"].dtype == torch.long
    assert outputs["codebook_indices"].dtype == torch.long
    assert outputs["backend_used"]["node_text_encoder"] == "hash_fallback"
    assert outputs["backend_used"]["passage_text_encoder"] == "hash_fallback"


def test_medical_graph_embedder_is_deterministic_for_same_bundle() -> None:
    bundle = make_evidence_bundle()
    embedder = build_embedder()

    first = embedder(bundle)
    second = embedder(bundle)

    assert torch.equal(first["inputs"], second["inputs"])
    assert torch.equal(first["puzzle_identifiers"], second["puzzle_identifiers"])
    assert torch.equal(first["codebook_indices"], second["codebook_indices"])
    assert first["node_mapping"] == second["node_mapping"]


def test_medical_graph_embedder_node_mapping_retains_provenance_ids() -> None:
    bundle = make_evidence_bundle()
    embedder = build_embedder()

    outputs = embedder(bundle)
    node_mapping = outputs["node_mapping"]

    assert len(node_mapping) == 256
    assert any(value == "drug:warfarin" for value in node_mapping.values())
    assert any(value == "PMID:12345678" for value in node_mapping.values())
    assert any(value == DEFAULT_PAD_IDENTIFIER for value in node_mapping.values())


def test_medical_graph_embedder_can_build_trm_input_bundle() -> None:
    bundle = make_evidence_bundle()
    embedder = build_embedder()

    trm_bundle = embedder.to_trm_input_bundle(bundle)

    assert trm_bundle.inputs.shape == (1, 256)
    assert trm_bundle.puzzle_identifiers.shape == (1,)
    assert trm_bundle.original_graph.question_id == bundle.question_id


def test_medical_graph_embedder_pretrain_vq_smoke() -> None:
    embedder = build_embedder()
    samples = [
        torch.linspace(0.0, 1.0, steps=64, dtype=torch.float32).reshape(1, 64),
        torch.linspace(1.0, 2.0, steps=64, dtype=torch.float32).reshape(1, 64),
    ]

    history = embedder.pretrain_vq(samples, epochs=1, lr=1e-3)

    assert len(history) == 1
    assert history[0]["loss"] >= 0.0
    assert history[0]["reconstruction_loss"] >= 0.0
    assert history[0]["commit_loss"] >= 0.0


def test_frozen_codebook_artifact_reload_is_token_stable(tmp_path) -> None:
    embedder = build_embedder()
    samples = [
        torch.linspace(float(index), float(index) + 1.0, steps=64, dtype=torch.float32).reshape(1, 64)
        for index in range(1000)
    ]

    history = embedder.pretrain_vq(samples, epochs=1, lr=1e-3)
    assert history[-1]["loss"] >= 0.0

    probe_vectors = torch.stack(
        [
            torch.linspace(0.1, 0.9, steps=64, dtype=torch.float32),
            torch.linspace(1.1, 1.9, steps=64, dtype=torch.float32),
            torch.linspace(2.1, 2.9, steps=64, dtype=torch.float32),
        ],
        dim=0,
    )
    usage_stats = inspect_codebook_usage(probe_vectors, embedder.vq.codebook.weight.detach().cpu())
    stability_before = estimate_token_stability(probe_vectors, embedder.vq.codebook.weight.detach().cpu(), repeats=2)
    metadata = embedder.build_codebook_metadata(num_subgraphs_seen=1000, seed=42)
    save_codebook_artifact(
        tmp_path,
        codebook_weight=embedder.vq.codebook.weight.detach().cpu(),
        metadata=metadata,
        stats={
            "usage_counts_total": usage_stats["usage_counts"],
            "top_used_clusters": usage_stats["top_used_clusters"],
            "unused_clusters": usage_stats["unused_clusters"][:10],
            "dead_code_ratio": usage_stats["dead_code_ratio"],
            "last_token_stability": stability_before,
        },
        progress={
            "status": "completed",
            "subgraphs_seen": 1000,
            "elapsed_sec": 0.0,
            "loss": history[-1]["loss"],
        },
        extra_state=embedder.export_codebook_state(),
    )

    reloaded = build_embedder()
    loaded_metadata = reloaded.load_frozen_codebook(tmp_path)
    first_ids = reloaded.vq(probe_vectors)[1]
    second_ids = reloaded.vq(probe_vectors)[1]

    assert loaded_metadata["codebook_version"] == metadata["codebook_version"]
    assert torch.equal(first_ids, second_ids)
