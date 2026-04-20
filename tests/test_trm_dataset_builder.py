from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.schemas import EvidenceBundle
from src.training import (
    TRMTrainingExample,
    build_trm_dataset_arrays,
    parse_reasoning_chain,
    path_to_token_sequence,
    save_trm_dataset,
)


class FakeEmbedder:
    padding_token_id = 4095

    def __call__(self, bundle: EvidenceBundle) -> dict[str, object]:
        del bundle
        inputs = torch.full((1, 256), self.padding_token_id, dtype=torch.long)
        inputs[0, :4] = torch.tensor([103, 101, 102, 104], dtype=torch.long)
        return {
            "inputs": inputs,
            "puzzle_identifiers": torch.tensor([0], dtype=torch.long),
            "node_mapping": {
                0: "protein:CYP2C9",
                1: "drug:warfarin",
                2: "drug:metronidazole",
                3: "drug:clarithromycin",
                **{index: "__pad__" for index in range(4, 256)},
            },
        }


def make_evidence() -> EvidenceBundle:
    return EvidenceBundle.model_validate(
        {
            "question_text": "Which H. pylori antibiotic has the highest warfarin bleeding risk?",
            "question_type": "drug_interaction",
            "question_entities": [
                {
                    "surface": "warfarin",
                    "cui": "C0043031",
                    "primekg_node_id": "drug:warfarin",
                    "entity_type": "drug",
                }
            ],
            "subgraph_edges": [
                {
                    "edge_id": "E_22104",
                    "head": "drug:warfarin",
                    "tail": "drug:metronidazole",
                    "relation": "drug_drug",
                    "display_relation": "interacts with",
                    "source_reliability": 0.95,
                    "amg_confidence": 1.0,
                    "supporting_pmids": ["28472901"],
                },
                {
                    "edge_id": "E_08812",
                    "head": "drug:metronidazole",
                    "tail": "protein:CYP2C9",
                    "relation": "drug_protein",
                    "display_relation": "targets",
                    "source_reliability": 0.9,
                    "amg_confidence": 1.0,
                    "supporting_pmids": ["28472901"],
                },
            ],
            "pubmed_passages": [],
            "metadata": {},
        }
    )


def test_parse_reasoning_chain_extracts_edge_ids() -> None:
    reasoning = [
        {"premise_edge_ids": ["edge:E_22104"]},
        "Metronidazole affects CYP2C9 [edge:E_08812].",
        {"claim_text": "backup edge_id:E_99999"},
    ]

    assert parse_reasoning_chain(reasoning) == ["E_22104", "E_08812", "E_99999"]


def test_path_to_token_sequence_maps_gold_edges_to_node_tokens() -> None:
    evidence = make_evidence()
    encoded = FakeEmbedder()(evidence)

    labels, diagnostics = path_to_token_sequence(["E_22104", "E_08812"], encoded, evidence)

    assert labels.shape == (256,)
    assert labels[:3].tolist() == [101, 102, 103]
    assert labels[3].item() == -100
    assert diagnostics["labeled_positions"] == [0, 1, 2]
    assert diagnostics["source_positions"] == [1, 2, 0]
    assert diagnostics["gold_path_len"] == 2
    assert diagnostics["labeled_token_count"] == 3
    assert diagnostics["label_mode"] == "ordered_node_token_sequence"
    assert diagnostics["missing_gold_edges"] == []
    assert diagnostics["missing_gold_nodes"] == []


def test_path_labels_are_not_trivial_position_copies() -> None:
    evidence = make_evidence()
    encoded = FakeEmbedder()(evidence)

    labels, _diagnostics = path_to_token_sequence(["E_22104", "E_08812"], encoded, evidence)
    inputs = encoded["inputs"][0]
    valid_mask = labels != -100

    assert labels[valid_mask].tolist() == [101, 102, 103]
    assert inputs[valid_mask].tolist() == [103, 101, 102]
    assert not torch.equal(labels[valid_mask], inputs[valid_mask])


def test_path_labels_change_with_gold_path_order_for_same_retrieval_order() -> None:
    evidence = make_evidence()
    encoded = FakeEmbedder()(evidence)

    forward_labels, forward_diagnostics = path_to_token_sequence(["E_22104", "E_08812"], encoded, evidence)
    reverse_labels, reverse_diagnostics = path_to_token_sequence(["E_08812", "E_22104"], encoded, evidence)

    assert forward_labels[:3].tolist() == [101, 102, 103]
    assert reverse_labels[:3].tolist() == [103, 102, 101]
    assert forward_diagnostics["ordered_gold_nodes"] == ["drug:warfarin", "drug:metronidazole", "protein:CYP2C9"]
    assert reverse_diagnostics["ordered_gold_nodes"] == ["protein:CYP2C9", "drug:metronidazole", "drug:warfarin"]


def test_build_and_save_trm_dataset_arrays(tmp_path: Path) -> None:
    evidence = make_evidence()
    arrays = build_trm_dataset_arrays(
        [TRMTrainingExample(evidence=evidence, gold_edge_ids=["E_22104", "E_08812"])],
        FakeEmbedder(),
    )

    assert arrays.inputs.shape == (1, 256)
    assert arrays.labels.shape == (1, 256)
    assert arrays.puzzle_identifiers.tolist() == [0]
    assert arrays.puzzle_indices.tolist() == [0, 1]
    assert arrays.group_indices.tolist() == [0, 1]
    assert arrays.metadata["seq_len"] == 256
    assert arrays.metadata["vocab_size"] == 8192
    assert arrays.metadata["pad_id"] == 4095
    assert arrays.metadata["label_stats"]["label_mode"] == "ordered_node_token_sequence"
    assert arrays.metadata["label_stats"]["rows_valid"] == 1
    assert arrays.metadata["label_stats"]["invalid_label_reject_count"] == 0
    assert arrays.row_metadata[0]["gold_path_len"] == 2
    assert arrays.row_metadata[0]["labeled_token_count"] == 3
    assert arrays.row_metadata[0]["missing_gold_nodes"] == []
    assert arrays.row_metadata[0]["missing_gold_edges"] == []

    save_dir = save_trm_dataset(arrays, tmp_path / "trm_medical", split="train")

    assert json.loads((save_dir / "dataset.json").read_text(encoding="utf-8"))["seq_len"] == 256
    assert json.loads((save_dir / "row_metadata.json").read_text(encoding="utf-8"))[0]["labeled_token_count"] == 3
    assert np.load(save_dir / "all__inputs.npy").shape == (1, 256)
    assert np.load(save_dir / "all__labels.npy").shape == (1, 256)
    assert np.load(save_dir / "all__puzzle_identifiers.npy").tolist() == [0]


def test_build_trm_dataset_rejects_empty_gold_path() -> None:
    evidence = make_evidence()

    arrays = build_trm_dataset_arrays([TRMTrainingExample(evidence=evidence, gold_edge_ids=[])], FakeEmbedder())

    assert arrays.inputs.shape == (0, 256)
    assert arrays.labels.shape == (0, 256)
    assert arrays.skipped_examples[0]["reason"] == "missing_gold_path"
    assert arrays.skipped_examples[0]["gold_path_len"] == 0
    assert arrays.skipped_examples[0]["labeled_token_count"] == 0
    assert arrays.metadata["label_stats"]["rows_valid"] == 0
    assert arrays.metadata["label_stats"]["invalid_label_reject_count"] == 1
    assert arrays.metadata["label_stats"]["invalid_label_reject_rate"] == 1.0
