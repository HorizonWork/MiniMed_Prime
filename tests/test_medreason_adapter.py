from __future__ import annotations

import json
from pathlib import Path

from src.schemas import EvidenceBundle
from src.training import (
    heuristic_map_reasoning_to_edges,
    load_medreason_records,
    map_medreason_edges,
    normalize_medreason_record,
    prepare_medreason_seed_jsonl,
    resolve_edge_mapper_config,
)


class FakeRetriever:
    def __init__(self, bundle: EvidenceBundle) -> None:
        self.bundle = bundle
        self.questions: list[str] = []

    def retrieve(self, question: str) -> EvidenceBundle:
        self.questions.append(question)
        return self.bundle


def make_bundle() -> EvidenceBundle:
    return EvidenceBundle.model_validate(
        {
            "question_text": "Which H pylori antibiotic interacts with warfarin?",
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
                    "amg_confidence": 0.9,
                    "supporting_pmids": ["28472901"],
                },
                {
                    "edge_id": "E_08812",
                    "head": "drug:metronidazole",
                    "tail": "protein:CYP2C9",
                    "relation": "drug_protein",
                    "display_relation": "targets",
                    "source_reliability": 0.9,
                    "amg_confidence": 0.9,
                    "supporting_pmids": ["28472901"],
                },
            ],
            "pubmed_passages": [
                {
                    "pmid": "28472901",
                    "title": "Warfarin antibiotic interaction",
                    "abstract": "Metronidazole interacts with warfarin through CYP2C9.",
                    "relevance_score": 10.0,
                }
            ],
            "metadata": {},
        }
    )


def test_normalize_medreason_record_extracts_question_options_and_edges() -> None:
    record = {
        "id": "mr-1",
        "question": "Which antibiotic has the highest bleeding risk?",
        "options": {"A": "Amoxicillin", "C": "Metronidazole"},
        "answer": "C",
        "reasoning": [{"premise_edge_ids": ["edge:E_22104"]}],
    }

    example = normalize_medreason_record(record)

    assert "Options:" in example.question
    assert example.answer == "C"
    assert example.direct_edge_ids == ["E_22104"]
    assert example.group_id == "mr-1"


def test_load_medreason_records_reads_local_jsonl(tmp_path: Path) -> None:
    dataset_path = tmp_path / "medreason.jsonl"
    dataset_path.write_text(json.dumps({"question": "Q", "answer": "A"}) + "\n", encoding="utf-8")

    records = load_medreason_records(dataset_path, split=None)

    assert records == [{"question": "Q", "answer": "A"}]


def test_load_medreason_records_applies_default_split_for_unsplit_local_file(tmp_path: Path) -> None:
    dataset_path = tmp_path / "medreason.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps({"question": f"Q{index}", "answer": "A"}) for index in range(20)) + "\n",
        encoding="utf-8",
    )

    train_records = load_medreason_records(dataset_path, split="train")
    validation_records = load_medreason_records(dataset_path, split="validation")
    test_records = load_medreason_records(dataset_path, split="test")

    assert len(train_records) == 18
    assert len(validation_records) == 1
    assert len(test_records) == 1
    assert train_records[-1]["question"] == "Q17"
    assert validation_records[0]["question"] == "Q18"
    assert test_records[0]["question"] == "Q19"


def test_heuristic_map_reasoning_to_edges_without_explicit_edge_ids() -> None:
    bundle = make_bundle()
    example = normalize_medreason_record(
        {
            "question": "Which H pylori antibiotic interacts with warfarin?",
            "answer": "Metronidazole",
            "reasoning": "Metronidazole interacts with warfarin and targets CYP2C9 according to PMID 28472901.",
        }
    )

    edge_ids = heuristic_map_reasoning_to_edges(example, bundle)

    assert edge_ids[:2] == ["E_22104", "E_08812"]


def test_map_medreason_edges_combines_direct_and_heuristic_edges() -> None:
    bundle = make_bundle()
    example = normalize_medreason_record(
        {
            "question": "Which H pylori antibiotic interacts with warfarin?",
            "answer": "Metronidazole",
            "reasoning": "Start from [edge:E_22104], then metronidazole targets CYP2C9.",
        }
    )

    edge_ids, diagnostics = map_medreason_edges(example=example, evidence=bundle, backend="heuristic")

    assert edge_ids == ["E_22104", "E_08812"]
    assert diagnostics.direct_edge_ids_in_evidence == ["E_22104"]
    assert diagnostics.backend_used == "heuristic"


def test_prepare_medreason_seed_jsonl_writes_evidence_and_gold_edges(tmp_path: Path) -> None:
    source = tmp_path / "medreason.jsonl"
    source.write_text(
        json.dumps(
                {
                    "question": "Which H pylori antibiotic interacts with warfarin?",
                    "answer": "Metronidazole",
                    "reasoning": "Metronidazole interacts with warfarin [edge:E_22104] and targets CYP2C9.",
                }
            )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "trm_seed.jsonl"

    result = prepare_medreason_seed_jsonl(
        source=source,
        output_jsonl=output_path,
        retriever=FakeRetriever(make_bundle()),  # type: ignore[arg-type]
        split=None,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert result.records_saved == 1
    assert payload["gold_edge_ids"] == ["E_22104", "E_08812"]
    assert payload["evidence"]["question_text"] == "Which H pylori antibiotic interacts with warfarin?"


def test_resolve_edge_mapper_auto_prefers_heuristic_without_llm(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    backend, model_name = resolve_edge_mapper_config("auto", "heuristic")

    assert backend == "heuristic"
    assert model_name == "heuristic"


def test_resolve_edge_mapper_auto_prefers_gemini_when_api_key_exists(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    backend, model_name = resolve_edge_mapper_config("auto", "heuristic")

    assert backend == "llm"
    assert model_name == "gemini-2.5-flash"


def test_resolve_edge_mapper_auto_prefers_openai_when_api_key_exists(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    backend, model_name = resolve_edge_mapper_config("auto", "heuristic")

    assert backend == "llm"
    assert model_name == "gpt-4o-mini"


def test_resolve_edge_mapper_auto_uses_explicit_llm_model(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    backend, model_name = resolve_edge_mapper_config("auto", "data/checkpoints/judges/medical_o1_verifier_3B")

    assert backend == "llm"
    assert model_name == "data/checkpoints/judges/medical_o1_verifier_3B"


def test_resolve_edge_mapper_normalizes_4omini_alias(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    backend, model_name = resolve_edge_mapper_config("auto", "4omini")

    assert backend == "llm"
    assert model_name == "gpt-4o-mini"
