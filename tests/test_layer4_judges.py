from __future__ import annotations

from src.layers.layer4_judges import (
    AnswerFaithfulnessGuardian,
    EntityValidationResult,
    EntityValidator,
    EvidenceGroundingInspector,
    HallucinationJudge,
    ReasoningSoundnessAuditor,
)
from src.schemas import EvidenceBundle, JudgeOutput, TRMOutput


def make_evidence_bundle() -> EvidenceBundle:
    return EvidenceBundle.model_validate(
        {
            "question_text": "Is ibuprofen unsafe with warfarin in atrial fibrillation?",
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
                    "display_relation": "interacts with",
                    "source_reliability": 0.95,
                    "amg_confidence": 0.9,
                    "supporting_pmids": ["12345678"],
                },
                {
                    "edge_id": "E2",
                    "head": "drug:warfarin",
                    "tail": "disease:af",
                    "relation": "indication",
                    "display_relation": "is indicated for",
                    "source_reliability": 0.92,
                    "amg_confidence": 0.88,
                    "supporting_pmids": ["23456789"],
                },
                {
                    "edge_id": "E3",
                    "head": "drug:ibuprofen",
                    "tail": "disease:bleeding",
                    "relation": "contraindication",
                    "display_relation": "increases bleeding risk in",
                    "source_reliability": 0.91,
                    "amg_confidence": 0.85,
                    "supporting_pmids": ["34567890"],
                },
            ],
            "pubmed_passages": [
                {
                    "pmid": "12345678",
                    "title": "Warfarin and ibuprofen interaction",
                    "abstract": "Ibuprofen increases bleeding risk when combined with warfarin.",
                    "relevance_score": 10.0,
                },
                {
                    "pmid": "23456789",
                    "title": "Warfarin in atrial fibrillation",
                    "abstract": "Warfarin is indicated for atrial fibrillation and reduces embolic stroke risk.",
                    "relevance_score": 9.0,
                },
            ],
            "metadata": {},
        }
    )


def make_trm_output() -> TRMOutput:
    return TRMOutput.model_validate(
        {
            "ranked_paths": [
                {
                    "nodes": ["drug:warfarin", "drug:ibuprofen", "disease:bleeding"],
                    "edges": ["E1", "E3"],
                    "confidence": 0.72,
                    "supporting_pmids": ["12345678", "34567890"],
                }
            ],
            "validity_score": 0.74,
            "contradiction_flag": False,
            "trace": [{"step": 1, "halted": True}],
        }
    )


def test_entity_validator_catches_fake_cui() -> None:
    bundle = make_evidence_bundle()
    validator = EntityValidator(model_name="heuristic", device="cpu")
    invalid_entities = list(bundle.question_entities)
    invalid_entities[0] = invalid_entities[0].model_copy(update={"cui": "C9999999"})

    result = validator.evaluate(question_text=bundle.question_text, entities=invalid_entities)

    assert result.valid is False
    assert any("C9999999" in item for item in result.mislinked)
    assert result.confidence < 1.0


def test_evidence_grounding_inspector_flags_unsupported_claim() -> None:
    bundle = make_evidence_bundle()
    inspector = EvidenceGroundingInspector(model_name="heuristic", device="cpu")

    output = inspector.evaluate(
        evidence_bundle=bundle,
        candidate_claims=["Aspirin cures pneumonia."],
    )

    assert output.claims[0].verdict == "unsupported"
    assert output.claims[0].severity >= 1
    assert output.h5_present is False


def test_reasoning_soundness_auditor_detects_skipped_reasoning_step() -> None:
    bundle = make_evidence_bundle()
    auditor = ReasoningSoundnessAuditor(model_name="heuristic", device="cpu")
    trm_output = TRMOutput.model_validate(
        {
            "ranked_paths": [
                {
                    "nodes": ["drug:warfarin", "drug:ibuprofen"],
                    "edges": ["E1"],
                    "confidence": 0.64,
                    "supporting_pmids": ["12345678"],
                }
            ],
            "validity_score": 0.7,
            "contradiction_flag": False,
            "trace": [{"step": 1}],
        }
    )

    result = auditor.evaluate(
        trm_output=trm_output,
        question="How does warfarin interact with ibuprofen?",
        evidence_bundle=bundle,
    )

    assert result.sound is False
    assert any("skips" in gap.lower() for gap in result.gaps)


def test_answer_faithfulness_guardian_abstains_on_h5() -> None:
    guardian = AnswerFaithfulnessGuardian(model_name="heuristic", device="cpu")
    entity_result = EntityValidationResult(
        valid=True,
        missing_entities=[],
        mislinked=[],
        confidence=1.0,
    )
    grounding_output = JudgeOutput.model_validate(
        {
            "claims": [
                {
                    "text": "Warfarin 500 mg daily with ibuprofen is safe.",
                    "claim_type": "life_critical",
                    "verdict": "unsupported",
                    "evidence_id": "none",
                    "severity": 3,
                    "rationale": "No supporting evidence exists for the claim.",
                }
            ],
            "faithfulness_score": 0.1,
            "h5_present": True,
            "overall_recommendation": "REVISE",
            "abstention_reason": None,
        }
    )

    result = guardian.evaluate(
        final_answer="Warfarin 500 mg daily with ibuprofen is safe.",
        entity_result=entity_result,
        grounding_output=grounding_output,
        soundness_output={"sound": True, "gaps": [], "rcs": 0.9, "rns": 0.9, "cdr": 0.9},
    )

    assert result.verdict == "ABSTAIN"
    assert result.h5_present is True


def test_hallucination_judge_routes_h5_to_abstain_and_tracks_parse_success() -> None:
    bundle = make_evidence_bundle()
    trm_output = make_trm_output()
    judge = HallucinationJudge(model_name="heuristic", device="cpu")

    result = judge.evaluate(
        final_answer="Warfarin 500 mg daily with ibuprofen is safe.",
        evidence_bundle=bundle,
        trm_output=trm_output,
    )

    assert result.final_recommendation == "ABSTAIN"
    assert result.h5_present is True
    assert result.parse_success_rate >= 0.95


def test_judge_base_prefers_openai_backend_when_api_key_and_remote_model(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    validator = EntityValidator(model_name="gpt-4o-mini", device="cpu")

    assert validator.backend == "openai"
