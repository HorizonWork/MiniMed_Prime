from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.layers.layer1_retrieval import PrimeKGExtractor
from src.schemas import EvidenceBundle, QuestionEntity
from src.training import (
    build_seed_evidence_bundle,
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

    def retrieve(self, question: str, question_type_hint: str | None = None) -> EvidenceBundle:
        del question_type_hint
        self.questions.append(question)
        return self.bundle


class StaticLinker:
    backend_used = "rule_based"

    def __init__(self, entities):
        self._entities = list(entities)

    def link(self, text: str):
        del text
        return [entity.model_copy(deep=True) for entity in self._entities]


class FailingLinker:
    backend_used = "should_not_be_used"

    def link(self, text: str):
        raise AssertionError(f"Legacy linker should not be used in data-only grounding modes: {text}")


class SilentPubMedRetriever:
    backend_used = "disabled"

    def retrieve(self, question: str, k: int = 0):
        del question, k
        return []


class GroundingAwareRetriever:
    def __init__(self, *, primekg_path: Path, entities=None, use_relation_filter: bool = False, linker=None) -> None:
        self.config = SimpleNamespace(
            primekg_path=primekg_path,
            pubmed_cache_path=primekg_path / "pubmed_cache.jsonl",
            primekg_top_k=10,
            pubmed_top_k=0,
            use_relation_filter=use_relation_filter,
            relation_filter=None,
            relation_filter_overrides=None,
        )
        self.linker = linker or StaticLinker(entities or [])
        self.kg_extractor = PrimeKGExtractor(primekg_path=primekg_path)
        self.pubmed = SilentPubMedRetriever()

    def retrieve(self, question: str, question_type_hint: str | None = None) -> EvidenceBundle:
        del question, question_type_hint
        raise AssertionError("retrieve() should not be used for data-only grounding modes.")


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


def write_grounding_primekg_tables(path: Path) -> None:
    (path / "nodes.csv").write_text(
        "\n".join(
            [
                "node_index,node_id,node_type,node_name,node_source",
                "1,disease:h_pylori,disease,Helicobacter pylori infectious disease,primekg",
                "2,disease:gastritis,disease,gastritis,primekg",
                "3,drug:warfarin,drug,warfarin,drugbank",
            ]
        ),
        encoding="utf-8",
    )
    (path / "edges.csv").write_text(
        "\n".join(
            [
                "relation,display_relation,x_index,y_index",
                "disease_disease,associated disease,1,2",
            ]
        ),
        encoding="utf-8",
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


def test_load_medreason_records_ignores_config_json_when_dataset_file_exists(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"builder_name": "medreason", "config_name": "default"}),
        encoding="utf-8",
    )
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"max_length": 512, "do_sample": False}),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 512}),
        encoding="utf-8",
    )
    (tmp_path / "ours_quality_33000.jsonl").write_text(
        json.dumps({"question": "Q1", "answer": "A1"}) + "\n",
        encoding="utf-8",
    )

    records = load_medreason_records(tmp_path, split=None)

    assert records == [{"question": "Q1", "answer": "A1"}]


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


def test_heuristic_map_reasoning_to_edges_does_not_use_answer_only_mentions() -> None:
    bundle = make_bundle()
    example = normalize_medreason_record(
        {
            "question": "Which antibiotic is the correct choice?",
            "answer": "Metronidazole",
            "reasoning": "The source cites PMID 28472901 but does not name any graph path.",
        }
    )

    edge_ids = heuristic_map_reasoning_to_edges(example, bundle)

    assert edge_ids == []


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

    payload = json.loads(result.output_path.read_text(encoding="utf-8").strip())
    assert result.records_saved == 1
    assert payload["gold_edge_ids"] == ["E_22104", "E_08812"]
    assert payload["evidence"]["question_text"] == "Which H pylori antibiotic interacts with warfarin?"


def test_build_seed_evidence_bundle_supports_deterministic_grounding_mode(tmp_path: Path) -> None:
    write_grounding_primekg_tables(tmp_path)
    retriever = GroundingAwareRetriever(
        primekg_path=tmp_path,
        entities=[QuestionEntity(surface="H pylori", cui=None, primekg_node_id=None, entity_type="disease")],
    )

    evidence = build_seed_evidence_bundle(
        question="Most sensitive test for H pylori is-",
        retriever=retriever,  # type: ignore[arg-type]
        grounding_mode="deterministic_v2",
    )

    assert evidence.metadata["grounding_mode"] == "deterministic_v2"
    assert evidence.metadata["grounding_backend"] == "deterministic_v2"
    assert evidence.metadata["seed_node_ids"] == ["disease:h_pylori"]
    assert evidence.metadata["unresolved_entities"] == []
    assert evidence.question_entities[0].primekg_node_id == "disease:h_pylori"


def test_build_seed_evidence_bundle_uses_lightweight_seed_extractor_instead_of_retriever_linker(tmp_path: Path) -> None:
    write_grounding_primekg_tables(tmp_path)
    retriever = GroundingAwareRetriever(
        primekg_path=tmp_path,
        linker=FailingLinker(),
    )

    evidence = build_seed_evidence_bundle(
        question="Most sensitive test for H pylori is-",
        retriever=retriever,  # type: ignore[arg-type]
        grounding_mode="deterministic_v2",
    )

    assert evidence.metadata["entity_linker_backend"] == "lightweight_seed_rules"
    assert evidence.metadata["seed_node_ids"] == ["disease:h_pylori"]


def test_prepare_medreason_seed_jsonl_records_grounding_metadata_for_deterministic_mode(tmp_path: Path) -> None:
    write_grounding_primekg_tables(tmp_path)
    source = tmp_path / "medreason.jsonl"
    source.write_text(
        json.dumps(
            {
                "question": "Most sensitive test for H pylori is-",
                "answer": "Biopsy urease test",
                "reasoning": "Helicobacter pylori infectious disease is associated with gastritis.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "trm_seed.jsonl"
    retriever = GroundingAwareRetriever(
        primekg_path=tmp_path,
        entities=[QuestionEntity(surface="H pylori", cui=None, primekg_node_id=None, entity_type="disease")],
    )

    result = prepare_medreason_seed_jsonl(
        source=source,
        output_jsonl=output_path,
        retriever=retriever,  # type: ignore[arg-type]
        split=None,
        grounding_mode="deterministic_v2",
        edge_mapper_backend="heuristic",
    )

    payload = json.loads(result.output_path.read_text(encoding="utf-8").strip())
    assert result.records_saved == 1
    assert payload["metadata"]["grounding_mode"] == "deterministic_v2"
    assert payload["metadata"]["grounding_backend"] == "deterministic_v2"
    assert payload["metadata"]["seed_node_ids"] == ["disease:h_pylori"]
    assert payload["metadata"]["unresolved_entities"] == []
    assert payload["metadata"]["candidate_stats"]["linked_count"] == 1


def test_prepare_medreason_seed_jsonl_records_llm_grounding_fallback_when_backend_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    write_grounding_primekg_tables(tmp_path)
    source = tmp_path / "medreason.jsonl"
    source.write_text(
        json.dumps(
            {
                "question": "Most sensitive test for H pylori is-",
                "answer": "Biopsy urease test",
                "reasoning": "Helicobacter pylori infectious disease is associated with gastritis.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "trm_seed.jsonl"
    retriever = GroundingAwareRetriever(
        primekg_path=tmp_path,
        entities=[QuestionEntity(surface="H pylori", cui=None, primekg_node_id=None, entity_type="disease")],
    )

    result = prepare_medreason_seed_jsonl(
        source=source,
        output_jsonl=output_path,
        retriever=retriever,  # type: ignore[arg-type]
        split=None,
        grounding_mode="llm_assisted_v1",
        edge_mapper_backend="heuristic",
        llm_model_name="heuristic",
    )

    payload = json.loads(result.output_path.read_text(encoding="utf-8").strip())
    assert result.records_saved == 1
    assert payload["metadata"]["grounding_mode"] == "llm_assisted_v1"
    assert payload["metadata"]["grounding_backend"] == "deterministic_v2"
    assert payload["metadata"]["llm_grounding_used"] is False
    assert payload["metadata"]["llm_grounding_fallback_reason"] == "llm_backend_unavailable"


def test_prepare_medreason_seed_jsonl_skipped_rows_keep_required_diagnostics(tmp_path: Path) -> None:
    source = tmp_path / "medreason.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "skip-1",
                "question": "Which antibiotic interacts with warfarin?",
                "answer": "Unknown",
                "reasoning": "The source cites a missing graph edge [edge:E_MISSING].",
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
        edge_mapper_backend="direct",
    )

    assert result.records_saved == 0
    assert result.records_skipped == 1
    assert result.skipped_path is not None
    assert result.skip_stage_counts == {"edge_mapping": 1}
    assert result.skip_reason_counts == {"no_gold_edges_mapped": 1}
    assert result.skipped_avg_edge_count == 2.0
    assert result.skipped_avg_entity_count == 1.0

    skipped_row = json.loads(result.skipped_path.read_text(encoding="utf-8").strip())
    required_fields = {
        "question",
        "question_type",
        "backend_used",
        "requested_backend",
        "entity_count",
        "linked_entities",
        "edge_count",
        "pubmed_count",
        "mapped_edge_ids",
        "missing_gold_edges",
        "retrieval_backend_summary",
        "skip_stage",
        "skip_reason",
    }
    assert required_fields.issubset(skipped_row)
    assert skipped_row["group_id"] == "skip-1"
    assert skipped_row["question"] == "Which antibiotic interacts with warfarin?"
    assert skipped_row["question_type"] == "drug_interaction"
    assert skipped_row["backend_used"] == "direct"
    assert skipped_row["requested_backend"] == "direct"
    assert skipped_row["entity_count"] == 1
    assert skipped_row["edge_count"] == 2
    assert skipped_row["pubmed_count"] == 1
    assert skipped_row["mapped_edge_ids"] == []
    assert skipped_row["missing_gold_edges"] == ["E_MISSING"]
    assert skipped_row["skip_stage"] == "edge_mapping"
    assert skipped_row["skip_reason"] == "no_gold_edges_mapped"
    assert skipped_row["reason"] == skipped_row["skip_reason"]
    assert skipped_row["diagnostics"]["missing_direct_edge_ids"] == ["E_MISSING"]


def test_prepare_medreason_seed_jsonl_skipped_rows_record_auto_backend_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    source = tmp_path / "medreason.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "skip-auto",
                "question": "Unrelated clinical note?",
                "answer": "No match",
                "reasoning": "No supplied reasoning edge can be grounded here.",
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
        edge_mapper_backend="auto",
    )

    assert result.records_saved == 0
    assert result.records_skipped == 1
    assert result.backend_used == "heuristic"
    assert result.skipped_path is not None
    skipped_row = json.loads(result.skipped_path.read_text(encoding="utf-8").strip())
    assert skipped_row["requested_backend"] == "auto"
    assert skipped_row["effective_backend"] == "heuristic"
    assert skipped_row["backend_used"] == "heuristic"
    assert skipped_row["skip_stage"] == "edge_mapping"
    assert skipped_row["skip_reason"] == "no_gold_edges_mapped"


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
