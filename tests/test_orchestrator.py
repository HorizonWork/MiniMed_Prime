from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import requests
import torch

from src.layers.layer4_judges import EntityValidationResult, FaithfulnessGuardianResult, ReasoningSoundnessResult
from src.orchestrator import MedicalReasoningSystemV3, SystemConfig
from src.schemas import AnswerWithTrace, EvidenceBundle, JudgeOutput, TRMOutput


def make_evidence_bundle(question_text: str = "Is ibuprofen unsafe with warfarin in atrial fibrillation?") -> EvidenceBundle:
    return EvidenceBundle.model_validate(
        {
            "question_text": question_text,
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
                    "abstract": "Warfarin is indicated for atrial fibrillation.",
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
                    "nodes": ["drug:warfarin", "drug:ibuprofen"],
                    "edges": ["E1"],
                    "confidence": 0.72,
                    "supporting_pmids": ["12345678"],
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


def make_grounding_output(
    recommendation: str = "ACCEPT",
    abstention_reason: str | None = None,
) -> JudgeOutput:
    return JudgeOutput.model_validate(
        {
            "claims": [
                {
                    "text": "warfarin interacts with ibuprofen.",
                    "claim_type": "factual",
                    "verdict": "supported",
                    "evidence_id": "edge:E1",
                    "severity": 0,
                    "rationale": "Supported by the KG edge.",
                }
            ],
            "faithfulness_score": 0.9,
            "h5_present": False,
            "overall_recommendation": recommendation,
            "abstention_reason": abstention_reason,
        }
    )


class ConstantJudge:
    def __init__(self, result: object) -> None:
        self.result = result

    def evaluate(self, **_: object) -> object:
        return self.result


class StubRetriever:
    def __init__(self, bundle: EvidenceBundle, *, raise_timeout: bool = False) -> None:
        self.bundle = bundle
        self.raise_timeout = raise_timeout
        self.config = SimpleNamespace(
            use_relation_filter=True,
            relation_filter=None,
            relation_filter_overrides=None,
            primekg_top_k=200,
            pubmed_cache_path=Path("data/pubmed_cache.jsonl"),
        )
        self.linker = SimpleNamespace(link=lambda question: list(self.bundle.question_entities))
        self.kg_extractor = SimpleNamespace(
            last_relation_filter_stats={},
            resolve_seed_entities=lambda entities: list(entities),
            extract_2hop=lambda **kwargs: list(self.bundle.subgraph_edges),
        )
        cached_articles = [
            SimpleNamespace(pmid=passage.pmid, title=passage.title, abstract=passage.abstract)
            for passage in self.bundle.pubmed_passages
        ]
        client = SimpleNamespace(
            _cache={f"{self.bundle.question_text.strip().lower()}::50": cached_articles},
            _cache_key=lambda query, retmax: f"{query.strip().lower()}::{retmax}",
            config=SimpleNamespace(search_retmax=50),
        )
        self.pubmed = SimpleNamespace(
            client=client,
            config=SimpleNamespace(search_candidate_count=50, top_k=5),
            _article_to_text=lambda article: f"{article.title} {article.abstract}".strip(),
            _bm25_scores=lambda query, documents: np.asarray([float(len(documents) - index) for index, _ in enumerate(documents)]),
        )

    def retrieve(self, question: str) -> EvidenceBundle:
        if self.raise_timeout:
            raise requests.Timeout("PubMed timeout")
        return self.bundle.model_copy(deep=True, update={"question_text": question})

    def _classify_question_type(self, question: str) -> str:
        del question
        return self.bundle.question_type


class StubEmbedder:
    def __init__(self, *, active_count: int = 4, padding_token_id: int = 4095) -> None:
        self.active_count = active_count
        self.padding_token_id = padding_token_id

    def __call__(self, evidence: EvidenceBundle) -> dict[str, object]:
        inputs = torch.full((1, 256), self.padding_token_id, dtype=torch.long)
        codebook_indices = torch.full((256,), self.padding_token_id, dtype=torch.long)
        node_mapping = {index: "__pad__" for index in range(256)}
        for index in range(self.active_count):
            inputs[0, index] = index + 1
            codebook_indices[index] = index + 1
            node_mapping[index] = evidence.question_entities[index % len(evidence.question_entities)].primekg_node_id
        return {
            "inputs": inputs,
            "puzzle_identifiers": torch.tensor([0], dtype=torch.long),
            "codebook_indices": codebook_indices,
            "node_mapping": node_mapping,
        }


class StubTRMReasoner:
    def __init__(self, output: TRMOutput, *, oom_on_first: bool = False) -> None:
        self.output = output
        self.oom_on_first = oom_on_first
        self.calls: list[dict[str, object]] = []

    def reason(self, embedder_output: dict[str, object]) -> TRMOutput:
        self.calls.append(embedder_output)
        active_count = len(
            [
                identifier
                for identifier in dict(embedder_output.get("node_mapping", {})).values()
                if identifier != "__pad__"
            ]
        )
        if self.oom_on_first and len(self.calls) == 1 and active_count > 8:
            raise RuntimeError("CUDA out of memory")
        return self.output


class StubSynthesizer:
    def __init__(self, answers: list[str] | None = None) -> None:
        self.answers = answers or [
            "Direct answer: Warfarin interacts with ibuprofen [edge:E1].\n"
            "Reasoning: Warfarin interacts with ibuprofen [edge:E1] [PMID:12345678].\n"
            "Confidence level: high (0.82) [edge:E1]."
        ]
        self.calls = 0
        self.last_clean_answer = ""
        self.last_provenance_issues: list[str] = []

    def synthesize(self, question: str, trm_output: TRMOutput, evidence: EvidenceBundle, judge_feedback: object | None = None) -> str:
        del question
        del trm_output
        del evidence
        del judge_feedback
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        self.last_clean_answer = answer
        return answer

    def enforce_provenance(self, answer: str, evidence: EvidenceBundle) -> tuple[bool, list[str]]:
        del evidence
        self.last_clean_answer = answer
        self.last_provenance_issues = []
        return True, []

    def format_abstention(self, reason: str, evidence_stats: dict[str, object]) -> str:
        tags = " ".join(evidence_stats.get("provenance_tags", ["[edge:E1]"]))
        score = float(evidence_stats.get("validity_score", 0.0))
        return (
            f"ABSTENTION: I cannot provide a confident answer because {reason.lower()} {tags}.\n"
            f"Reasoning: The evidence bundle remains insufficient {tags}.\n"
            f"Confidence level: low ({score:.2f}) {tags}."
        )


class StubHallucinationJudge:
    def __init__(self, results: list[object]) -> None:
        self.results = results
        self.calls = 0

    def evaluate(self, *, final_answer: str, evidence_bundle: EvidenceBundle, trm_output: TRMOutput, candidate_claims: list[str] | None = None) -> object:
        del final_answer
        del evidence_bundle
        del trm_output
        del candidate_claims
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return result


def build_system(
    *,
    bundle: EvidenceBundle | None = None,
    trm_output: TRMOutput | None = None,
    retriever: object | None = None,
    embedder: object | None = None,
    trm: object | None = None,
    synthesizer: object | None = None,
    hallucination_results: list[object] | None = None,
    preflight_grounding: JudgeOutput | None = None,
) -> MedicalReasoningSystemV3:
    bundle = bundle or make_evidence_bundle()
    trm_output = trm_output or make_trm_output()
    preflight_grounding = preflight_grounding or make_grounding_output()

    judges_component = {
        "entity": ConstantJudge(
            EntityValidationResult(valid=True, missing_entities=[], mislinked=[], confidence=1.0)
        ),
        "grounding": ConstantJudge(preflight_grounding),
        "soundness": ConstantJudge(
            ReasoningSoundnessResult(sound=True, gaps=[], rcs=0.9, rns=0.9, cdr=0.9)
        ),
        "faithfulness": ConstantJudge(
            FaithfulnessGuardianResult(
                verdict="ACCEPT",
                reasoning="All checks passed.",
                confidence=0.95,
                h5_present=False,
            )
        ),
    }

    hallucination_results = hallucination_results or [
        SimpleNamespace(
            final_recommendation="ACCEPT",
            grounding_output=make_grounding_output(),
            abstention_reason=None,
            confidence=0.92,
            parse_success_rate=1.0,
        )
    ]

    config = SystemConfig(
        retriever_component=retriever or StubRetriever(bundle),
        embedder_component=embedder or StubEmbedder(),
        trm_component=trm or StubTRMReasoner(trm_output),
        judges_component=judges_component,
        synthesizer_component=synthesizer or StubSynthesizer(),
        hallucination_judge_component=StubHallucinationJudge(hallucination_results),
    )
    return MedicalReasoningSystemV3(config)


def test_orchestrator_happy_path_returns_answer_with_trace() -> None:
    system = build_system()

    result = system.answer("Is ibuprofen unsafe with warfarin in atrial fibrillation?")

    assert isinstance(result, AnswerWithTrace)
    assert result.abstention is False
    assert result.provenance == ["edge:E1", "PMID:12345678"]
    assert result.judge_outputs
    assert result.trm_trace is not None
    assert result.trm_trace[-1]["stage"] == "orchestrator"
    assert result.trm_trace[-1]["provenance"] == result.provenance


def test_orchestrator_abstains_when_trm_confidence_is_low() -> None:
    system = build_system(trm_output=make_trm_output(validity_score=0.35))

    result = system.answer("Is ibuprofen unsafe with warfarin in atrial fibrillation?")

    assert result.abstention is True
    assert result.abstention_reason == "TRM confidence too low"
    assert result.provenance


def test_orchestrator_retries_synthesis_on_revise_then_accepts() -> None:
    synthesizer = StubSynthesizer(
        answers=[
            "Direct answer: Warfarin interacts with ibuprofen [edge:E1].\n"
            "Reasoning: Warfarin interacts with ibuprofen [edge:E1].\n"
            "Confidence level: moderate (0.82) [edge:E1].",
            "Direct answer: Evidence suggests warfarin interacts with ibuprofen [edge:E1].\n"
            "Reasoning: Warfarin interacts with ibuprofen [edge:E1] [PMID:12345678].\n"
            "Confidence level: moderate (0.82) [edge:E1].",
        ]
    )
    revised_grounding = make_grounding_output(recommendation="REVISE")
    system = build_system(
        synthesizer=synthesizer,
        hallucination_results=[
            SimpleNamespace(
                final_recommendation="REVISE",
                grounding_output=revised_grounding,
                abstention_reason=None,
                confidence=0.7,
                parse_success_rate=1.0,
            ),
            SimpleNamespace(
                final_recommendation="ACCEPT",
                grounding_output=make_grounding_output(),
                abstention_reason=None,
                confidence=0.9,
                parse_success_rate=1.0,
            ),
        ],
    )

    result = system.answer("Is ibuprofen unsafe with warfarin in atrial fibrillation?")

    assert result.abstention is False
    assert synthesizer.calls == 2
    assert "Evidence suggests" in result.answer_text


def test_orchestrator_retries_trm_after_oom_with_reduced_support() -> None:
    trm_reasoner = StubTRMReasoner(make_trm_output(), oom_on_first=True)
    embedder = StubEmbedder(active_count=12)
    system = build_system(embedder=embedder, trm=trm_reasoner)

    result = system.answer("Is ibuprofen unsafe with warfarin in atrial fibrillation?")

    assert result.abstention is False
    assert len(trm_reasoner.calls) == 2
    second_call_mapping = dict(trm_reasoner.calls[1]["node_mapping"])
    retained_nodes = [identifier for identifier in second_call_mapping.values() if identifier != "__pad__"]
    assert len(retained_nodes) == 8


def test_orchestrator_uses_cached_pubmed_on_timeout() -> None:
    bundle = make_evidence_bundle()
    retriever = StubRetriever(bundle, raise_timeout=True)
    system = build_system(bundle=bundle, retriever=retriever)

    result = system.answer(bundle.question_text)

    assert result.abstention is False
    assert system.last_state is not None
    assert system.last_state.evidence is not None
    assert system.last_state.evidence.metadata["pubmed_cache_fallback"] is True
