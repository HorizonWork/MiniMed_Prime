from __future__ import annotations

import re

from src.layers.layer5_synthesis import AnswerSynthesizer
from src.schemas import EvidenceBundle, JudgeOutput, TRMOutput


TAG_PATTERN = re.compile(r"\[(edge:[^\]]+|PMID:[^\]]+)\]")


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
                    "abstract": "Warfarin is indicated for atrial fibrillation and reduces embolic stroke risk at 5 mg/day in selected cohorts.",
                    "relevance_score": 9.0,
                },
            ],
            "metadata": {},
        }
    )


def make_trm_output(validity_score: float = 0.82, contradiction_flag: bool = False) -> TRMOutput:
    return TRMOutput.model_validate(
        {
            "ranked_paths": [
                {
                    "nodes": ["drug:warfarin", "drug:ibuprofen", "disease:bleeding"],
                    "edges": ["E1", "E3"],
                    "confidence": 0.72,
                    "supporting_pmids": ["12345678", "34567890"],
                },
                {
                    "nodes": ["drug:warfarin", "disease:af"],
                    "edges": ["E2"],
                    "confidence": 0.81,
                    "supporting_pmids": ["23456789"],
                },
            ],
            "validity_score": validity_score,
            "contradiction_flag": contradiction_flag,
            "trace": [{"step": 1, "halted": True}],
        }
    )


def split_sentences(text: str) -> list[str]:
    return [fragment.strip() for fragment in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if fragment.strip()]


def allowed_refs(bundle: EvidenceBundle) -> set[str]:
    return {f"edge:{edge.edge_id}" for edge in bundle.subgraph_edges} | {
        f"PMID:{passage.pmid}" for passage in bundle.pubmed_passages
    }


def test_synthesize_tags_every_sentence_and_uses_only_known_references() -> None:
    bundle = make_evidence_bundle()
    trm_output = make_trm_output()
    synthesizer = AnswerSynthesizer(model_path="heuristic")

    answer = synthesizer.synthesize(bundle.question_text, trm_output, bundle)

    sentences = split_sentences(answer)
    refs = TAG_PATTERN.findall(answer)

    assert sentences
    assert all(TAG_PATTERN.search(sentence) for sentence in sentences)
    assert set(refs).issubset(allowed_refs(bundle))
    assert synthesizer.last_backend_used == "heuristic"


def test_enforce_provenance_drops_invalid_or_missing_sentences() -> None:
    bundle = make_evidence_bundle()
    synthesizer = AnswerSynthesizer(model_path="heuristic")
    answer = (
        "Supported statement [edge:E1]. "
        "Hallucinated statement [PMID:99999999]. "
        "Missing provenance statement."
    )

    is_clean, issues = synthesizer.enforce_provenance(answer, bundle)

    assert is_clean is False
    assert len(issues) == 2
    assert "Supported statement [edge:E1]." in synthesizer.last_clean_answer
    assert "PMID:99999999" not in synthesizer.last_clean_answer
    assert "Missing provenance statement" not in synthesizer.last_clean_answer


def test_synthesize_abstains_when_validity_score_is_below_threshold() -> None:
    bundle = make_evidence_bundle()
    trm_output = make_trm_output(validity_score=0.35)
    synthesizer = AnswerSynthesizer(model_path="heuristic")

    answer = synthesizer.synthesize(bundle.question_text, trm_output, bundle)

    assert answer.startswith("ABSTENTION:")
    assert all(TAG_PATTERN.search(sentence) for sentence in split_sentences(answer))


def test_synthesize_uses_uncertainty_phrasing_for_mid_confidence() -> None:
    bundle = make_evidence_bundle()
    trm_output = make_trm_output(validity_score=0.55)
    synthesizer = AnswerSynthesizer(model_path="heuristic")

    answer = synthesizer.synthesize(bundle.question_text, trm_output, bundle)

    assert "Evidence suggests" in answer


def test_synthesize_abstains_on_judge_abstention() -> None:
    bundle = make_evidence_bundle()
    trm_output = make_trm_output(validity_score=0.82)
    judge_feedback = JudgeOutput.model_validate(
        {
            "claims": [
                {
                    "text": "Ibuprofen is fully safe with warfarin.",
                    "claim_type": "life_critical",
                    "verdict": "unsupported",
                    "evidence_id": "none",
                    "severity": 3,
                    "rationale": "The evidence does not support this safety claim.",
                }
            ],
            "faithfulness_score": 0.2,
            "h5_present": True,
            "overall_recommendation": "ABSTAIN",
            "abstention_reason": "Blocking H5 issue.",
        }
    )
    synthesizer = AnswerSynthesizer(model_path="heuristic")

    answer = synthesizer.synthesize(bundle.question_text, trm_output, bundle, judge_feedback=judge_feedback)

    assert answer.startswith("ABSTENTION:")


def test_auto_backend_prefers_gguf_directory(monkeypatch, tmp_path) -> None:
    model_dir = tmp_path / "llm"
    model_dir.mkdir()
    (model_dir / "II-Medical-Q4_K_M.gguf").write_bytes(b"gguf")

    def fake_load_gguf_backend(self: AnswerSynthesizer) -> bool:
        self.backend = "llama_cpp"
        self.last_backend_used = self.backend
        self.model = object()
        return True

    def fake_load_transformers_backend(self: AnswerSynthesizer) -> bool:
        raise AssertionError("transformers backend should not be used when gguf is present")

    monkeypatch.setattr(AnswerSynthesizer, "_load_gguf_backend", fake_load_gguf_backend)
    monkeypatch.setattr(AnswerSynthesizer, "_load_transformers_backend", fake_load_transformers_backend)

    synthesizer = AnswerSynthesizer(model_path=str(model_dir), backend="auto")

    assert synthesizer.backend == "llama_cpp"
    assert synthesizer.last_backend_used == "llama_cpp"


def test_auto_backend_uses_transformers_when_no_gguf(monkeypatch, tmp_path) -> None:
    model_dir = tmp_path / "transformers-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    def fake_load_gguf_backend(self: AnswerSynthesizer) -> bool:
        raise AssertionError("gguf backend should not be used when no gguf file exists")

    def fake_load_transformers_backend(self: AnswerSynthesizer) -> bool:
        self.backend = "transformers_4bit"
        self.last_backend_used = self.backend
        self.model = object()
        self.tokenizer = object()
        return True

    monkeypatch.setattr(AnswerSynthesizer, "_load_gguf_backend", fake_load_gguf_backend)
    monkeypatch.setattr(AnswerSynthesizer, "_load_transformers_backend", fake_load_transformers_backend)

    synthesizer = AnswerSynthesizer(model_path=str(model_dir), backend="auto")

    assert synthesizer.backend == "transformers_4bit"
    assert synthesizer.last_backend_used == "transformers_4bit"


def test_auto_backend_uses_openai_when_api_key_and_remote_model(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    synthesizer = AnswerSynthesizer(model_path="gpt-4o-mini", backend="auto")

    assert synthesizer.backend == "openai"
    assert synthesizer.last_backend_used == "openai"
