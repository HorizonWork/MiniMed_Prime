from __future__ import annotations

import torch
import pytest
from pydantic import ValidationError

from src.schemas import (
    AnswerWithTrace,
    ClaimVerdict,
    DEFAULT_ABSTENTION_REASON_BLOCKING_CLAIM,
    DEFAULT_ABSTENTION_REASON_H5,
    EvidenceBundle,
    JudgeOutput,
    KGEdge,
    Path,
    PubMedPassage,
    QuestionEntity,
    TRMInputBundle,
    TRMOutput,
)


def make_question_entity_payload() -> dict[str, object]:
    return {
        "surface": "aspirin",
        "cui": "C0004057",
        "primekg_node_id": "drug:aspirin",
        "entity_type": "drug",
    }


def make_kg_edge_payload() -> dict[str, object]:
    return {
        "edge_id": "E-100",
        "head": "drug:aspirin",
        "tail": "disease:stroke",
        "relation": "indication",
        "display_relation": "indication",
        "source_reliability": 0.92,
        "amg_confidence": 0.88,
        "supporting_pmids": ["12345678", "23456789"],
    }


def make_pubmed_passage_payload() -> dict[str, object]:
    return {
        "pmid": "12345678",
        "title": "Aspirin in stroke prevention",
        "abstract": "Aspirin reduces recurrent vascular events in selected populations.",
        "relevance_score": 17.4,
    }


def make_evidence_bundle_payload() -> dict[str, object]:
    return {
        "question_text": "Can aspirin help prevent recurrent stroke?",
        "question_type": "factoid",
        "question_entities": [make_question_entity_payload()],
        "subgraph_edges": [make_kg_edge_payload()],
        "pubmed_passages": [make_pubmed_passage_payload()],
        "metadata": {"retriever": "layer1"},
    }


def make_path_payload() -> dict[str, object]:
    return {
        "nodes": ["drug:aspirin", "disease:stroke"],
        "edges": ["E-100"],
        "confidence": 0.81,
        "supporting_pmids": ["12345678"],
    }


def make_trm_output_payload() -> dict[str, object]:
    return {
        "ranked_paths": [make_path_payload()],
        "validity_score": 0.79,
        "contradiction_flag": False,
        "trace": [{"step": 1, "decision": "expand"}],
    }


def make_claim_verdict_payload() -> dict[str, object]:
    return {
        "text": "Aspirin is indicated for recurrent stroke prevention.",
        "claim_type": "factual",
        "verdict": "supported",
        "evidence_id": "edge_id:E-100",
        "severity": 1,
        "rationale": "The KG edge and literature passage support this association.",
    }


def make_judge_output_payload() -> dict[str, object]:
    return {
        "claims": [make_claim_verdict_payload()],
        "faithfulness_score": 0.9,
        "h5_present": False,
        "overall_recommendation": "ACCEPT",
        "abstention_reason": "should be cleared",
    }


def make_answer_with_trace_payload() -> dict[str, object]:
    evidence = EvidenceBundle.model_validate(make_evidence_bundle_payload())
    return {
        "answer_text": "Aspirin can help prevent recurrent stroke in appropriate patients. [edge:E-100] [PMID:12345678]",
        "question_id": evidence.question_id,
        "provenance": ["edge_id:E-100", "PMID:12345678"],
        "confidence": 0.87,
        "abstention": False,
        "abstention_reason": None,
        "judge_outputs": [make_judge_output_payload()],
        "trm_trace": [{"step": 1, "confidence": 0.87}],
    }


def test_model_validate_public_models() -> None:
    question_entity = QuestionEntity.model_validate(make_question_entity_payload())
    kg_edge = KGEdge.model_validate(make_kg_edge_payload())
    pubmed_passage = PubMedPassage.model_validate(make_pubmed_passage_payload())
    evidence_bundle = EvidenceBundle.model_validate(make_evidence_bundle_payload())
    reasoning_path = Path.model_validate(make_path_payload())
    trm_output = TRMOutput.model_validate(make_trm_output_payload())
    claim_verdict = ClaimVerdict.model_validate(make_claim_verdict_payload())
    judge_output = JudgeOutput.model_validate(make_judge_output_payload())
    answer_with_trace = AnswerWithTrace.model_validate(make_answer_with_trace_payload())
    trm_input = TRMInputBundle.model_validate(
        {
            "inputs": [list(range(256))],
            "puzzle_identifiers": [7],
            "original_graph": make_evidence_bundle_payload(),
        }
    )

    assert question_entity.entity_type == "drug"
    assert kg_edge.relation == "indication"
    assert pubmed_passage.pmid == "12345678"
    assert evidence_bundle.question_id
    assert reasoning_path.edges == ["E-100"]
    assert trm_output.ranked_paths[0].confidence == pytest.approx(0.81)
    assert claim_verdict.evidence_id == "edge:E-100"
    assert judge_output.abstention_reason is None
    assert answer_with_trace.provenance == ["edge:E-100", "PMID:12345678"]
    assert isinstance(trm_input.inputs, torch.Tensor)
    assert trm_input.inputs.dtype == torch.long


def test_evidence_bundle_json_round_trip_is_lossless() -> None:
    evidence_bundle = EvidenceBundle.model_validate(make_evidence_bundle_payload())
    round_tripped = EvidenceBundle.model_validate_json(evidence_bundle.model_dump_json())

    assert round_tripped.model_dump() == evidence_bundle.model_dump()


def test_trm_input_bundle_tensor_passthrough() -> None:
    inputs = torch.arange(256, dtype=torch.long).reshape(1, 256)
    puzzle_identifiers = torch.tensor([5], dtype=torch.long)

    bundle = TRMInputBundle.model_validate(
        {
            "inputs": inputs,
            "puzzle_identifiers": puzzle_identifiers,
            "original_graph": make_evidence_bundle_payload(),
        }
    )

    assert torch.equal(bundle.inputs, inputs)
    assert torch.equal(bundle.puzzle_identifiers, puzzle_identifiers)


def test_trm_input_bundle_serializes_tensors_as_lists() -> None:
    bundle = TRMInputBundle.model_validate(
        {
            "inputs": [list(range(256))],
            "puzzle_identifiers": [3],
            "original_graph": make_evidence_bundle_payload(),
        }
    )

    dumped = bundle.model_dump(mode="json")

    assert dumped["inputs"] == [list(range(256))]
    assert dumped["puzzle_identifiers"] == [3]


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("inputs", [list(range(255))]),
        ("inputs", [[1, 2], [3, 4]]),
        ("puzzle_identifiers", []),
        ("puzzle_identifiers", [[7]]),
    ],
)
def test_trm_input_bundle_rejects_invalid_shapes(field_name: str, field_value: object) -> None:
    payload = {
        "inputs": [list(range(256))],
        "puzzle_identifiers": [7],
        "original_graph": make_evidence_bundle_payload(),
    }
    payload[field_name] = field_value

    with pytest.raises(ValidationError):
        TRMInputBundle.model_validate(payload)


def test_invalid_primekg_relation_is_rejected() -> None:
    payload = make_kg_edge_payload()
    payload["relation"] = "not_a_real_relation"

    with pytest.raises(ValidationError):
        KGEdge.model_validate(payload)


@pytest.mark.parametrize("evidence_id", ["bad", "edge:", "PMID:", "edge: bad"])
def test_invalid_claim_evidence_formats_are_rejected(evidence_id: str) -> None:
    payload = make_claim_verdict_payload()
    payload["evidence_id"] = evidence_id

    with pytest.raises(ValidationError):
        ClaimVerdict.model_validate(payload)


@pytest.mark.parametrize("provenance", [["bad"], ["edge:"], ["PMID:"], ["edge: bad"]])
def test_invalid_answer_provenance_formats_are_rejected(provenance: list[str]) -> None:
    payload = make_answer_with_trace_payload()
    payload["provenance"] = provenance

    with pytest.raises(ValidationError):
        AnswerWithTrace.model_validate(payload)


@pytest.mark.parametrize("severity", [-1, 4])
def test_invalid_severity_is_rejected(severity: int) -> None:
    payload = make_claim_verdict_payload()
    payload["severity"] = severity

    with pytest.raises(ValidationError):
        ClaimVerdict.model_validate(payload)


def test_inconsistent_path_lengths_are_rejected() -> None:
    payload = make_path_payload()
    payload["edges"] = []

    with pytest.raises(ValidationError):
        Path.model_validate(payload)


def test_judge_output_auto_abstains_on_blocking_claim() -> None:
    claim_payload = make_claim_verdict_payload()
    claim_payload["severity"] = 3

    judge_output = JudgeOutput.model_validate(
        {
            "claims": [claim_payload],
            "faithfulness_score": 0.5,
            "h5_present": False,
            "overall_recommendation": "ACCEPT",
            "abstention_reason": None,
        }
    )

    assert judge_output.overall_recommendation == "ABSTAIN"
    assert judge_output.abstention_reason == DEFAULT_ABSTENTION_REASON_BLOCKING_CLAIM


def test_judge_output_auto_abstains_on_h5() -> None:
    judge_output = JudgeOutput.model_validate(
        {
            "claims": [make_claim_verdict_payload()],
            "faithfulness_score": 0.7,
            "h5_present": True,
            "overall_recommendation": "REVISE",
            "abstention_reason": None,
        }
    )

    assert judge_output.overall_recommendation == "ABSTAIN"
    assert judge_output.abstention_reason == DEFAULT_ABSTENTION_REASON_H5


def test_claim_verdict_normalizes_legacy_edge_ids() -> None:
    claim_verdict = ClaimVerdict.model_validate(make_claim_verdict_payload())

    assert claim_verdict.evidence_id == "edge:E-100"


def test_claim_verdict_allows_none_evidence_id() -> None:
    payload = make_claim_verdict_payload()
    payload["evidence_id"] = "none"

    claim_verdict = ClaimVerdict.model_validate(payload)

    assert claim_verdict.evidence_id == "none"
