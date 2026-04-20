from __future__ import annotations

import json
from pathlib import Path

from judges.posthoc_auditor import PostHocAuditor
from src.schemas import EvidenceBundle
from tools.run_posthoc_audit import main as run_posthoc_audit_main


def _bundle() -> EvidenceBundle:
    return EvidenceBundle.model_validate(
        {
            "question_text": "Is ibuprofen contraindicated with warfarin?",
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
            ],
            "subgraph_edges": [
                {
                    "edge_id": "E1",
                    "head": "drug:warfarin",
                    "tail": "drug:ibuprofen",
                    "relation": "drug_drug",
                    "display_relation": "interacts with",
                    "source_reliability": 0.95,
                    "amg_confidence": 0.90,
                    "supporting_pmids": ["12345678"],
                },
                {
                    "edge_id": "E2",
                    "head": "drug:ibuprofen",
                    "tail": "disease:bleeding",
                    "relation": "contraindication",
                    "display_relation": "increases bleeding risk in",
                    "source_reliability": 0.91,
                    "amg_confidence": 0.89,
                    "supporting_pmids": ["12345678"],
                },
            ],
            "pubmed_passages": [
                {
                    "pmid": "12345678",
                    "title": "Ibuprofen and warfarin interaction",
                    "abstract": "Ibuprofen increases bleeding risk when combined with warfarin.",
                    "relevance_score": 10.0,
                }
            ],
            "metadata": {},
        }
    )


def test_posthoc_auditor_flags_fake_source_references() -> None:
    auditor = PostHocAuditor(model_name="heuristic", device="cpu")

    result = auditor.evaluate_with_diagnostics(
        question="Is ibuprofen contraindicated with warfarin?",
        evidence_bundle=_bundle(),
        candidate_answer="Metronidazole is safest [edge:E99999] [PMID:99999999].",
        mode="answer_only",
    )

    assert "edge:E99999" in result.output.unsupported_evidence_references
    assert "PMID:99999999" in result.output.unsupported_evidence_references
    assert any(claim.type == "sourceattribution" for claim in result.output.claims)
    assert result.output.overall_recommendation in {"REVISE", "ABSTAIN"}


def test_posthoc_auditor_abstains_on_life_critical_claim() -> None:
    auditor = PostHocAuditor(model_name="heuristic", device="cpu")

    result = auditor.evaluate_with_diagnostics(
        question="Is ibuprofen contraindicated with warfarin?",
        evidence_bundle=_bundle(),
        candidate_answer="Warfarin and ibuprofen are safe together at 500 mg daily.",
        mode="answer_only",
    )

    assert result.output.h5_present is True
    assert result.output.overall_recommendation == "ABSTAIN"
    assert result.output.sample_label == "reject"


def test_posthoc_auditor_fallback_sets_revise_after_invalid_json() -> None:
    class StubBrokenAuditor(PostHocAuditor):
        def __init__(self) -> None:
            self._outputs = ["not json", "[]", "invalid {json"]
            super().__init__(model_name="gpt-4o-mini", device="cpu")

        def _load_backend(self) -> None:
            self.backend = "stub"

        def _generate_structured_text(self, prompt: str) -> str:
            del prompt
            if self._outputs:
                return self._outputs.pop(0)
            return "not json"

    auditor = StubBrokenAuditor()
    result = auditor.evaluate_with_diagnostics(
        question="Is ibuprofen contraindicated with warfarin?",
        evidence_bundle=_bundle(),
        candidate_answer="Ibuprofen and warfarin can interact.",
        mode="answer_only",
    )

    assert result.parse_success is False
    assert result.json_retry_count == 3
    assert result.output.overall_recommendation == "REVISE"
    assert result.output.sample_label == "partial"


def test_run_posthoc_audit_from_trace_file_writes_artifacts(tmp_path: Path) -> None:
    trace_path = tmp_path / "sample.trace.json"
    trace_payload = {
        "sample_id": "sample-42",
        "question": "Is ibuprofen contraindicated with warfarin?",
        "gold_answer": "Ibuprofen increases bleeding risk with warfarin [edge:E1] [PMID:12345678].",
        "gold_reasoning": "Warfarin interacts with ibuprofen and increases bleeding risk.",
        "question_type": {"predicted": "drug_interaction"},
        "entity_extraction": {
            "entities": [
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
            ]
        },
        "primekg_retrieval": {
            "top_edges": [
                {
                    "edge_id": "E1",
                    "head": "drug:warfarin",
                    "tail": "drug:ibuprofen",
                    "relation": "drug_drug",
                    "display_relation": "interacts with",
                    "source_reliability": 0.95,
                    "amg_confidence": 0.90,
                    "supporting_pmids": ["12345678"],
                }
            ]
        },
        "pubmed_retrieval": {
            "passages": [
                {
                    "pmid": "12345678",
                    "title": "Ibuprofen and warfarin interaction",
                    "abstract": "Ibuprofen increases bleeding risk when combined with warfarin.",
                    "relevance_score": 10.0,
                }
            ]
        },
    }
    trace_path.write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")
    output_dir = tmp_path / "audits"

    exit_code = run_posthoc_audit_main(
        [
            "--trace-file",
            str(trace_path),
            "--output-dir",
            str(output_dir),
            "--mode",
            "answer_plus_cot",
            "--auditor-model",
            "heuristic",
        ]
    )

    assert exit_code == 0
    audit_json = output_dir / "sample-42.audit.json"
    audit_md = output_dir / "sample-42.audit.md"
    assert audit_json.exists()
    assert audit_md.exists()
    payload = json.loads(audit_json.read_text(encoding="utf-8"))
    assert payload["sample_id"] == "sample-42"
    assert payload["overall_recommendation"] in {"ACCEPT", "REVISE", "ABSTAIN"}
    assert isinstance(payload["claims"], list)
