from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests

from src.layers import (
    AgenticRetriever,
    AnswerFaithfulnessGuardian,
    AnswerSynthesizer,
    EntityValidator,
    EvidenceGroundingInspector,
    HallucinationJudge,
    MedicalGraphEmbedder,
    ReasoningSoundnessAuditor,
    TRMReasoner,
)
from src.schemas import AnswerWithTrace, EvidenceBundle, JudgeOutput, PubMedPassage, TRMOutput
from src.utils.kaggle_env import KaggleEnv
from src.utils.path_resolver import KagglePathResolver
from src.utils.structured_logger import DEFAULT_LOG_DIR, StructuredLogger

try:
    import torch
except ImportError:  # pragma: no cover - torch is part of the project stack
    torch = None

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())


DEFAULT_CONF_THRESHOLD = 0.4
DEFAULT_SYNTHESIS_RETRY = 2
DEFAULT_TRM_RETRY_SUPPORT_LIMIT = 8
DEFAULT_DEFAULT_JUDGE_MODEL = "heuristic"
DEFAULT_DEFAULT_SYNTHESIS_MODEL = "heuristic"
DEFAULT_DEFAULT_TRM_MODEL = "medical_trm.ckpt"
DEFAULT_PIPELINE_TRACE_VERSION = "medical_reasoning_v3"
DEFAULT_PAD_IDENTIFIER = "__pad__"
DEFAULT_PROVENANCE_PATTERN = re.compile(r"\[(edge:[^\]]+|edge_id:[^\]]+|PMID:[^\]]+)\]")
DEFAULT_TRM_REPO_PATH = "external/TinyRecursiveModels"
DEFAULT_KAGGLE_TRM_PATH = "/kaggle/input/medv3-checkpoints/trm"
DEFAULT_SAPBERT_MODEL_PATH = "data/checkpoints/sapbert"
DEFAULT_KAGGLE_SAPBERT_MODEL_PATH = "/kaggle/input/medv3-checkpoints/sapbert"
DEFAULT_MEDCPT_ARTICLE_MODEL_PATH = "data/checkpoints/medcpt-article"
DEFAULT_SYNTHESIS_MODEL_PATH = "data/checkpoints/medreason-8b"


def _default_existing_path(path_str: str) -> str | None:
    resolved = KaggleEnv.path(path_str)
    return str(resolved) if resolved.exists() else None


def _first_existing_path(candidates: list[str | Path]) -> str | None:
    for candidate in candidates:
        path = candidate if isinstance(candidate, Path) else Path(candidate)
        resolved = path if path.is_absolute() else KaggleEnv.path(path)
        if resolved.exists():
            return str(resolved)
    return None


def _default_synthesis_model_path() -> str:
    resolved = KaggleEnv.path(DEFAULT_SYNTHESIS_MODEL_PATH)
    if resolved.exists():
        return str(resolved)
    return DEFAULT_SYNTHESIS_MODEL_PATH


def _default_trm_model_path() -> str:
    checkpoint_path = KagglePathResolver.resolve_trm_checkpoint()
    if checkpoint_path:
        return checkpoint_path
    repo_root = KagglePathResolver.resolve_trm_repo()
    if repo_root:
        return repo_root
    resolved = KaggleEnv.path(DEFAULT_TRM_REPO_PATH)
    if resolved.exists():
        return str(resolved)
    return DEFAULT_DEFAULT_TRM_MODEL


def _default_sapbert_model_path() -> str | None:
    return KagglePathResolver.resolve_sapbert() or _first_existing_path([DEFAULT_KAGGLE_SAPBERT_MODEL_PATH, DEFAULT_SAPBERT_MODEL_PATH])


def _default_sapbert_path() -> str | None:
    return KagglePathResolver.resolve_sapbert() or _first_existing_path([DEFAULT_KAGGLE_SAPBERT_MODEL_PATH, DEFAULT_SAPBERT_MODEL_PATH])


def _default_trm_repo_path() -> str | None:
    return KagglePathResolver.resolve_trm_repo()


def _default_trm_checkpoint_path() -> str | None:
    return KagglePathResolver.resolve_trm_checkpoint()


@dataclass(slots=True)
class SystemConfig:
    primekg_path: Path = KaggleEnv.path("data/kg/primekg")
    pubmed_cache_path: Path = KaggleEnv.ensure_writeable(KaggleEnv.path("data/pubmed_cache.jsonl"))
    pubmed_api_key: str | None = None
    scispacy_model: str = "en_core_sci_lg"
    use_relation_filter: bool = True
    relation_filter: set[str] | None = None
    relation_filter_overrides: dict[str, set[str]] | None = None

    embedder_hidden_dim: int = 256
    embedder_codebook_size: int = 4096
    embedder_max_len: int = 256
    embedder_n_relations: int = 30
    embedder_n_node_types: int = 10
    sapbert_path: str | None = field(default_factory=_default_sapbert_path)
    sapbert_model_name: str | None = field(default_factory=_default_sapbert_model_path)
    medcpt_article_model_name: str | None = field(default_factory=lambda: _default_existing_path(DEFAULT_MEDCPT_ARTICLE_MODEL_PATH))
    embedder_device: str = "cpu"
    embedder_frozen_codebook_path: str | None = None
    embedder_allow_codebook_fallback: bool = True
    embedder_strict_source_graph_signature: bool = False

    trm_model_path: str = field(default_factory=_default_trm_model_path)
    trm_repo_path: str | None = field(default_factory=_default_trm_repo_path)
    trm_checkpoint_path: str | None = field(default_factory=_default_trm_checkpoint_path)
    trm_device: str = "cpu"
    trm_max_steps: int = 16
    trm_conf_threshold: float = 0.85
    trm_contradiction_threshold: float = 0.5
    trm_retry_support_limit: int = DEFAULT_TRM_RETRY_SUPPORT_LIMIT
    trm_strict_validate: bool = True

    judge_model_name: str = DEFAULT_DEFAULT_JUDGE_MODEL
    judge_device: str = "cpu"
    judge_max_retries: int = 2

    synthesis_model_path: str = field(default_factory=_default_synthesis_model_path)
    synthesis_n_ctx: int = 8192

    conf_threshold: float = DEFAULT_CONF_THRESHOLD
    synthesis_max_retry: int = DEFAULT_SYNTHESIS_RETRY

    retriever_component: Any | None = None
    embedder_component: Any | None = None
    trm_component: Any | None = None
    judges_component: dict[str, Any] | None = None
    synthesizer_component: Any | None = None
    hallucination_judge_component: Any | None = None


@dataclass(slots=True)
class PipelineState:
    question: str
    evidence: EvidenceBundle | None = None
    embedder_output: dict[str, Any] | None = None
    trm_output: TRMOutput | None = None
    preflight_judge_output: JudgeOutput | None = None
    final_judge_output: JudgeOutput | None = None
    answer_text: str | None = None
    errors: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


class MedicalReasoningSystemV3:
    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self.event_logger = StructuredLogger("orchestrator", DEFAULT_LOG_DIR)
        resolved_sapbert_path = config.sapbert_path or KagglePathResolver.resolve_sapbert()
        resolved_trm_checkpoint = config.trm_checkpoint_path or KagglePathResolver.resolve_trm_checkpoint()
        resolved_trm_repo = config.trm_repo_path or KagglePathResolver.resolve_trm_repo()
        resolved_trm_model_path = resolved_trm_checkpoint or resolved_trm_repo or config.trm_model_path
        resolved_paths = {
            "sapbert_path": resolved_sapbert_path,
            "trm_checkpoint_path": resolved_trm_checkpoint,
            "trm_repo_path": resolved_trm_repo,
            "trm_model_path": resolved_trm_model_path,
        }
        logger.info("Orchestrator paths: %s", resolved_paths)
        self.config.sapbert_path = resolved_sapbert_path
        self.config.trm_checkpoint_path = resolved_trm_checkpoint
        self.config.trm_repo_path = resolved_trm_repo
        self.config.trm_model_path = resolved_trm_model_path

        self.retriever = config.retriever_component or AgenticRetriever(
            primekg_path=config.primekg_path,
            pubmed_api_key=config.pubmed_api_key,
            pubmed_cache_path=config.pubmed_cache_path,
            scispacy_model=config.scispacy_model,
            use_relation_filter=config.use_relation_filter,
            relation_filter=config.relation_filter,
            relation_filter_overrides=config.relation_filter_overrides,
        )
        self.embedder = config.embedder_component or MedicalGraphEmbedder(
            hidden_dim=config.embedder_hidden_dim,
            codebook_size=config.embedder_codebook_size,
            max_len=config.embedder_max_len,
            n_relations=config.embedder_n_relations,
            n_node_types=config.embedder_n_node_types,
            sapbert_path=resolved_sapbert_path,
            sapbert_model_name=config.sapbert_model_name or resolved_sapbert_path,
            medcpt_article_model_name=config.medcpt_article_model_name,
            device=config.embedder_device,
            primekg_path=config.primekg_path,
            frozen_codebook_path=config.embedder_frozen_codebook_path,
            allow_codebook_fallback=config.embedder_allow_codebook_fallback,
            strict_source_graph_signature=config.embedder_strict_source_graph_signature,
        )
        self.trm = config.trm_component or TRMReasoner(
            model_path=resolved_trm_model_path,
            device=config.trm_device,
            max_steps=config.trm_max_steps,
            conf_threshold=config.trm_conf_threshold,
            contradiction_threshold=config.trm_contradiction_threshold,
            strict_validate=config.trm_strict_validate,
        )

        if config.judges_component is not None:
            self.judges = dict(config.judges_component)
        else:
            self.judges = {
                "entity": EntityValidator(model_name=config.judge_model_name, device=config.judge_device),
                "grounding": EvidenceGroundingInspector(model_name=config.judge_model_name, device=config.judge_device),
                "soundness": ReasoningSoundnessAuditor(model_name=config.judge_model_name, device=config.judge_device),
                "faithfulness": AnswerFaithfulnessGuardian(model_name=config.judge_model_name, device=config.judge_device),
            }

        self.synthesizer = config.synthesizer_component or AnswerSynthesizer(
            model_path=config.synthesis_model_path,
            n_ctx=config.synthesis_n_ctx,
        )

        self.hallucination_judge = config.hallucination_judge_component or HallucinationJudge(
            model_name=config.judge_model_name,
            device=config.judge_device,
            max_retries=config.judge_max_retries,
            abstain_threshold=config.conf_threshold,
        )
        self._sync_wrapper_judges()

        self.conf_threshold = config.conf_threshold
        self.last_state: PipelineState | None = None

    def answer(self, question: str) -> AnswerWithTrace:
        state = PipelineState(question=question)
        self.last_state = state
        self._record_event(state, "pipeline_start", {"question": question})
        self.event_logger.log_event(
            "pipeline_start",
            {"question": question, "backend_used": "medical_reasoning_v3", "latency_ms": 0.0},
        )

        try:
            evidence = self._retrieve_with_fallback(question=question, state=state)
            state.evidence = evidence

            embedder_output = self._embed(evidence=evidence, state=state)
            state.embedder_output = embedder_output

            trm_output = self._reason_with_retry(embedder_output=embedder_output, state=state)
            state.trm_output = trm_output

            if trm_output.validity_score < self.conf_threshold:
                return self._abstain(
                    evidence=evidence,
                    trm_output=trm_output,
                    reason="TRM confidence too low",
                    judge_outputs=[],
                    state=state,
                )

            preflight_judge_output = self._run_preflight_judges(
                question=question,
                evidence=evidence,
                trm_output=trm_output,
                state=state,
            )
            state.preflight_judge_output = preflight_judge_output

            if preflight_judge_output.overall_recommendation == "ABSTAIN":
                return self._abstain(
                    evidence=evidence,
                    trm_output=trm_output,
                    reason=preflight_judge_output.abstention_reason or "Judge recommended abstention",
                    judge_outputs=[preflight_judge_output],
                    state=state,
                )

            answer_text, final_grounding_output = self._synthesize_with_retry(
                question=question,
                trm_out=trm_output,
                evidence=evidence,
                judge_out=preflight_judge_output,
                state=state,
                max_retry=self.config.synthesis_max_retry,
            )
            state.answer_text = answer_text
            state.final_judge_output = final_grounding_output

            provenance = self._extract_provenance(answer_text)
            judge_outputs = self._deduplicate_judge_outputs([preflight_judge_output, final_grounding_output])
            trace = self._build_trace(state=state, trm_output=trm_output, provenance=provenance)

            return AnswerWithTrace(
                answer_text=answer_text,
                question_id=evidence.question_id,
                provenance=provenance,
                confidence=trm_output.validity_score,
                abstention=False,
                abstention_reason=None,
                judge_outputs=judge_outputs,
                trm_trace=trace,
            )
        except PipelineAbstention as abstention:
            if state.evidence is not None and state.trm_output is not None:
                return self._abstain(
                    evidence=state.evidence,
                    trm_output=state.trm_output,
                    reason=abstention.reason,
                    judge_outputs=abstention.judge_outputs,
                    state=state,
                )
            logger.exception("Pipeline abstained before evidence or TRM output was available.")
            self.event_logger.log_exception(
                "pipeline_abstention_exception",
                abstention,
                {"question": question, "backend_used": "medical_reasoning_v3", "latency_ms": 0.0},
            )
            return self._emergency_abstain(question=question, reason=abstention.reason, state=state)
        except Exception as exc:
            logger.exception(f"Unhandled pipeline exception for question '{question}': {exc}")
            state.errors.append(str(exc))
            self.event_logger.log_exception(
                "pipeline_exception",
                exc,
                {"question": question, "backend_used": "medical_reasoning_v3", "latency_ms": 0.0},
            )
            return self._emergency_abstain(question=question, reason=str(exc), state=state)

    def _abstain(
        self,
        *,
        evidence: EvidenceBundle,
        trm_output: TRMOutput,
        reason: str,
        judge_outputs: list[JudgeOutput],
        state: PipelineState | None = None,
    ) -> AnswerWithTrace:
        evidence_stats = {
            "validity_score": trm_output.validity_score,
            "edge_count": len(evidence.subgraph_edges),
            "passage_count": len(evidence.pubmed_passages),
            "provenance_tags": [
                f"[{reference}]" for reference in self._fallback_provenance(evidence=evidence)
            ],
        }
        answer_text = self.synthesizer.format_abstention(reason=reason, evidence_stats=evidence_stats)
        self.synthesizer.enforce_provenance(answer_text, evidence)
        clean_answer = self.synthesizer.last_clean_answer or answer_text
        provenance = self._extract_provenance(clean_answer)
        if not provenance:
            provenance = self._fallback_provenance(evidence=evidence)
        if state is not None:
            state.answer_text = clean_answer
            self._record_event(state, "abstain", {"reason": reason, "provenance": provenance})
        self.event_logger.log_event(
            "pipeline_abstain",
            {
                "reason": reason,
                "provenance": provenance,
                "backend_used": "medical_reasoning_v3",
                "latency_ms": 0.0,
            },
        )
        trace = self._build_trace(state=state, trm_output=trm_output, provenance=provenance)
        return AnswerWithTrace(
            answer_text=clean_answer,
            question_id=evidence.question_id,
            provenance=provenance,
            confidence=trm_output.validity_score,
            abstention=True,
            abstention_reason=reason,
            judge_outputs=self._deduplicate_judge_outputs(judge_outputs),
            trm_trace=trace,
        )

    def _synthesize_with_retry(
        self,
        question: str,
        trm_out: TRMOutput,
        evidence: EvidenceBundle,
        judge_out: JudgeOutput,
        state: PipelineState,
        max_retry: int = DEFAULT_SYNTHESIS_RETRY,
    ) -> tuple[str, JudgeOutput]:
        current_feedback = judge_out
        last_grounding_output = judge_out
        for attempt_index in range(max_retry + 1):
            answer_text = self.synthesizer.synthesize(
                question=question,
                trm_output=trm_out,
                evidence=evidence,
                judge_feedback=current_feedback,
            )
            self._record_event(
                state,
                "synthesis_attempt",
                {
                    "attempt": attempt_index + 1,
                    "answer_preview": answer_text[:240],
                    "judge_recommendation": current_feedback.overall_recommendation,
                    "backend_used": getattr(self.synthesizer, "last_backend_used", getattr(self.synthesizer, "backend", None)),
                },
            )

            hallucination_result = self._evaluate_synthesized_answer(
                final_answer=answer_text,
                evidence=evidence,
                trm_output=trm_out,
                state=state,
            )
            last_grounding_output = hallucination_result.grounding_output
            self._record_event(
                state,
                "post_synthesis_judging",
                {
                    "attempt": attempt_index + 1,
                    "final_recommendation": hallucination_result.final_recommendation,
                    "confidence": hallucination_result.confidence,
                    "parse_success_rate": hallucination_result.parse_success_rate,
                },
            )

            if hallucination_result.final_recommendation == "ABSTAIN":
                raise PipelineAbstention(
                    reason=hallucination_result.abstention_reason or "Judge recommended abstention after synthesis",
                    judge_outputs=self._deduplicate_judge_outputs([judge_out, last_grounding_output]),
                )

            if hallucination_result.final_recommendation == "ACCEPT":
                return answer_text, last_grounding_output

            current_feedback = last_grounding_output

        return answer_text, last_grounding_output

    def _retrieve_with_fallback(self, *, question: str, state: PipelineState) -> EvidenceBundle:
        try:
            evidence = self.retriever.retrieve(question)
            self._record_event(
                state,
                "retrieval_complete",
                {
                    "question_id": evidence.question_id,
                    "edge_count": len(evidence.subgraph_edges),
                    "pubmed_count": len(evidence.pubmed_passages),
                    "metadata": dict(evidence.metadata),
                },
            )
            return evidence
        except Exception as exc:
            if not self._is_timeout_error(exc):
                raise
            logger.warning(f"PubMed retrieval timed out for '{question}'; falling back to local cache.")
            self.event_logger.log_exception(
                "retrieval_timeout_fallback",
                exc,
                {"question": question, "backend_used": "medical_reasoning_v3", "latency_ms": 0.0},
            )
            evidence = self._retrieve_from_cache(question=question)
            self._record_event(
                state,
                "retrieval_cache_fallback",
                {
                    "question_id": evidence.question_id,
                    "edge_count": len(evidence.subgraph_edges),
                    "pubmed_count": len(evidence.pubmed_passages),
                },
            )
            return evidence

    def _retrieve_from_cache(self, *, question: str) -> EvidenceBundle:
        question_type = self.retriever._classify_question_type(question)
        question_entities = self.retriever.linker.link(question)
        question_entities = self.retriever.kg_extractor.resolve_seed_entities(question_entities)
        subgraph_edges = self.retriever.kg_extractor.extract_2hop(
            seed_entities=question_entities,
            top_k=self.retriever.config.primekg_top_k,
            relation_filter=self.retriever.config.relation_filter,
            question_type=question_type,
            use_relation_filter=bool(getattr(self.retriever.config, "use_relation_filter", True)),
            relation_filter_overrides=getattr(self.retriever.config, "relation_filter_overrides", None),
        )
        pubmed_passages = self._cached_pubmed_passages(question)
        metadata = {
            "question_type": question_type,
            "pubmed_cache_fallback": True,
            "entity_count": len(question_entities),
            "edge_count": len(subgraph_edges),
            "pubmed_count": len(pubmed_passages),
            "pubmed_cache_path": str(self.retriever.config.pubmed_cache_path),
            "relation_filter": dict(getattr(self.retriever.kg_extractor, "last_relation_filter_stats", {})),
        }
        return EvidenceBundle(
            question_text=question,
            question_type=question_type,
            question_entities=question_entities,
            subgraph_edges=subgraph_edges,
            pubmed_passages=pubmed_passages,
            metadata=metadata,
        )

    def _cached_pubmed_passages(self, question: str) -> list[PubMedPassage]:
        pubmed_retriever = getattr(self.retriever, "pubmed", None)
        client = getattr(pubmed_retriever, "client", None)
        if client is None:
            return []

        search_retmax = getattr(getattr(pubmed_retriever, "config", None), "search_candidate_count", None)
        if search_retmax is None:
            search_retmax = getattr(getattr(client, "config", None), "search_retmax", 50)

        cache_lookup = getattr(client, "_cache", {})
        cache_key = None
        if hasattr(client, "_cache_key"):
            cache_key = client._cache_key(query=question, retmax=search_retmax)
        articles = list(cache_lookup.get(cache_key, [])) if cache_key is not None else []

        if not articles:
            normalized_prefix = f"{question.strip().lower()}::"
            for candidate_key, candidate_articles in cache_lookup.items():
                if candidate_key.startswith(normalized_prefix):
                    articles = list(candidate_articles)
                    break

        if not articles:
            return []

        document_texts = [pubmed_retriever._article_to_text(article) for article in articles]
        bm25_scores = pubmed_retriever._bm25_scores(query=question, documents=document_texts)
        ranked_indices = list(reversed(bm25_scores.argsort()))[: getattr(pubmed_retriever.config, "top_k", 5)]

        passages: list[PubMedPassage] = []
        for index in ranked_indices:
            article = articles[int(index)]
            passages.append(
                PubMedPassage(
                    pmid=article.pmid,
                    title=article.title,
                    abstract=article.abstract,
                    relevance_score=float(bm25_scores[int(index)]),
                )
            )
        return passages

    def _embed(self, *, evidence: EvidenceBundle, state: PipelineState) -> dict[str, Any]:
        embedder_output = dict(self.embedder(evidence))
        embedder_output.setdefault("original_graph", evidence)
        self._record_event(
            state,
            "embedding_complete",
            {
                "input_shape": tuple(getattr(embedder_output.get("inputs"), "shape", ())),
                "puzzle_identifiers_shape": tuple(getattr(embedder_output.get("puzzle_identifiers"), "shape", ())),
                "backend_used": embedder_output.get("backend_used"),
            },
        )
        return embedder_output

    def _reason_with_retry(self, *, embedder_output: dict[str, Any], state: PipelineState) -> TRMOutput:
        try:
            trm_output = self.trm.reason(embedder_output)
            self._record_event(
                state,
                "trm_reasoning_complete",
                {
                    "validity_score": trm_output.validity_score,
                    "contradiction_flag": trm_output.contradiction_flag,
                    "path_count": len(trm_output.ranked_paths),
                    "backend_used": getattr(self.trm, "backend_used", None),
                    "trm_is_real": getattr(self.trm, "trm_is_real", False),
                },
            )
            return trm_output
        except RuntimeError as exc:
            if not self._is_oom_error(exc):
                raise
            logger.warning(
                f"TRM OOM encountered; reducing active support to {self.config.trm_retry_support_limit} and retrying."
            )
            self.event_logger.log_exception(
                "trm_oom_retry",
                exc,
                {"support_limit": self.config.trm_retry_support_limit, "backend_used": getattr(self.trm, "backend_used", None), "latency_ms": 0.0},
            )
            reduced_output = self._reduce_support_tokens(embedder_output=embedder_output, keep_count=self.config.trm_retry_support_limit)
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            trm_output = self.trm.reason(reduced_output)
            self._record_event(
                state,
                "trm_reasoning_retry",
                {
                    "validity_score": trm_output.validity_score,
                    "contradiction_flag": trm_output.contradiction_flag,
                    "support_limit": self.config.trm_retry_support_limit,
                    "backend_used": getattr(self.trm, "backend_used", None),
                    "trm_is_real": getattr(self.trm, "trm_is_real", False),
                },
            )
            return trm_output

    def _run_preflight_judges(
        self,
        *,
        question: str,
        evidence: EvidenceBundle,
        trm_output: TRMOutput,
        state: PipelineState,
    ) -> JudgeOutput:
        try:
            entity_result = self.judges["entity"].evaluate(
                question_text=question,
                entities=evidence.question_entities,
            )
            candidate_claims = self._claims_from_trm_paths(evidence=evidence, trm_output=trm_output)
            grounding_output = self.judges["grounding"].evaluate(
                evidence_bundle=evidence,
                candidate_claims=candidate_claims,
            )
            soundness_output = self.judges["soundness"].evaluate(
                trm_output=trm_output,
                question=question,
                evidence_bundle=evidence,
            )
            faithfulness_output = self.judges["faithfulness"].evaluate(
                final_answer=" ".join(candidate_claims),
                entity_result=entity_result,
                grounding_output=grounding_output,
                soundness_output=soundness_output,
            )
            if faithfulness_output.verdict == "ABSTAIN":
                grounding_output = grounding_output.model_copy(
                    update={
                        "overall_recommendation": "ABSTAIN",
                        "abstention_reason": grounding_output.abstention_reason or faithfulness_output.reasoning,
                    }
                )
            self._record_event(
                state,
                "preflight_judges_complete",
                {
                    "candidate_claim_count": len(candidate_claims),
                    "entity_valid": entity_result.valid,
                    "sound": soundness_output.sound,
                    "recommendation": grounding_output.overall_recommendation,
                    "faithfulness_verdict": faithfulness_output.verdict,
                },
            )
            return grounding_output
        except Exception as exc:
            logger.warning(f"Judge execution failed; falling back to deterministic heuristic judges: {exc}")
            self.event_logger.log_exception(
                "judge_fallback",
                exc,
                {"backend_used": "deterministic_judges", "latency_ms": 0.0},
            )
            deterministic_judges = self._build_deterministic_judges()
            entity_result = deterministic_judges["entity"].evaluate(question_text=question, entities=evidence.question_entities)
            candidate_claims = self._claims_from_trm_paths(evidence=evidence, trm_output=trm_output)
            grounding_output = deterministic_judges["grounding"].evaluate(
                evidence_bundle=evidence,
                candidate_claims=candidate_claims,
            )
            soundness_output = deterministic_judges["soundness"].evaluate(
                trm_output=trm_output,
                question=question,
                evidence_bundle=evidence,
            )
            faithfulness_output = deterministic_judges["faithfulness"].evaluate(
                final_answer=" ".join(candidate_claims),
                entity_result=entity_result,
                grounding_output=grounding_output,
                soundness_output=soundness_output,
            )
            if faithfulness_output.verdict == "ABSTAIN":
                grounding_output = grounding_output.model_copy(
                    update={
                        "overall_recommendation": "ABSTAIN",
                        "abstention_reason": grounding_output.abstention_reason or faithfulness_output.reasoning,
                    }
                )
            self._record_event(
                state,
                "preflight_judges_fallback",
                {
                    "candidate_claim_count": len(candidate_claims),
                    "recommendation": grounding_output.overall_recommendation,
                },
            )
            return grounding_output

    def _evaluate_synthesized_answer(
        self,
        *,
        final_answer: str,
        evidence: EvidenceBundle,
        trm_output: TRMOutput,
        state: PipelineState,
    ) -> Any:
        try:
            return self.hallucination_judge.evaluate(
                final_answer=final_answer,
                evidence_bundle=evidence,
                trm_output=trm_output,
            )
        except Exception as exc:
            logger.warning(f"HallucinationJudge failed; falling back to heuristic wrapper: {exc}")
            self.event_logger.log_exception(
                "hallucination_judge_fallback",
                exc,
                {"backend_used": "heuristic_wrapper", "latency_ms": 0.0},
            )
            fallback_wrapper = HallucinationJudge(
                model_name="heuristic",
                device="cpu",
                max_retries=self.config.judge_max_retries,
                abstain_threshold=self.conf_threshold,
            )
            self._record_event(state, "hallucination_judge_fallback", {"reason": str(exc)})
            return fallback_wrapper.evaluate(
                final_answer=final_answer,
                evidence_bundle=evidence,
                trm_output=trm_output,
            )

    def _build_deterministic_judges(self) -> dict[str, Any]:
        return {
            "entity": EntityValidator(model_name="heuristic", device="cpu"),
            "grounding": EvidenceGroundingInspector(model_name="heuristic", device="cpu"),
            "soundness": ReasoningSoundnessAuditor(model_name="heuristic", device="cpu"),
            "faithfulness": AnswerFaithfulnessGuardian(model_name="heuristic", device="cpu"),
        }

    def _sync_wrapper_judges(self) -> None:
        if hasattr(self.hallucination_judge, "entity_validator") and "entity" in self.judges:
            self.hallucination_judge.entity_validator = self.judges["entity"]
        if hasattr(self.hallucination_judge, "grounding_inspector") and "grounding" in self.judges:
            self.hallucination_judge.grounding_inspector = self.judges["grounding"]
        if hasattr(self.hallucination_judge, "soundness_auditor") and "soundness" in self.judges:
            self.hallucination_judge.soundness_auditor = self.judges["soundness"]
        if hasattr(self.hallucination_judge, "faithfulness_guardian") and "faithfulness" in self.judges:
            self.hallucination_judge.faithfulness_guardian = self.judges["faithfulness"]

    def _claims_from_trm_paths(self, *, evidence: EvidenceBundle, trm_output: TRMOutput) -> list[str]:
        edge_lookup = {edge.edge_id: edge for edge in evidence.subgraph_edges}
        claims: list[str] = []
        for path in trm_output.ranked_paths:
            for edge_id in path.edges:
                edge = edge_lookup.get(edge_id)
                if edge is None:
                    continue
                claims.append(f"{self._humanize_identifier(edge.head)} {edge.display_relation.lower()} {self._humanize_identifier(edge.tail)}.")
        if not claims:
            claims.append(f"Retrieved evidence is relevant to {evidence.question_text}.")
        deduplicated_claims = list(dict.fromkeys(claims))
        return deduplicated_claims[:5]

    def _reduce_support_tokens(self, *, embedder_output: dict[str, Any], keep_count: int) -> dict[str, Any]:
        reduced = dict(embedder_output)
        inputs = embedder_output.get("inputs")
        node_mapping = dict(embedder_output.get("node_mapping", {}))
        padding_token_id = getattr(self.embedder, "padding_token_id", None)

        if inputs is None or torch is None or not isinstance(inputs, torch.Tensor):
            return reduced

        reduced_inputs = inputs.clone()
        if reduced_inputs.ndim != 2:
            return reduced

        if padding_token_id is None:
            padding_token_id = int(reduced_inputs[0, -1].item())

        active_positions = [
            index
            for index, identifier in sorted(node_mapping.items())
            if identifier != DEFAULT_PAD_IDENTIFIER
        ]
        for index in active_positions[keep_count:]:
            reduced_inputs[:, index] = padding_token_id
            node_mapping[index] = DEFAULT_PAD_IDENTIFIER

        reduced["inputs"] = reduced_inputs
        if "codebook_indices" in reduced and isinstance(reduced["codebook_indices"], torch.Tensor):
            codebook_indices = reduced["codebook_indices"].clone()
            for index in active_positions[keep_count:]:
                codebook_indices[index] = padding_token_id
            reduced["codebook_indices"] = codebook_indices
        reduced["node_mapping"] = node_mapping
        return reduced

    def _extract_provenance(self, answer_text: str) -> list[str]:
        provenance: list[str] = []
        for raw_reference in DEFAULT_PROVENANCE_PATTERN.findall(answer_text):
            normalized_reference = raw_reference
            if raw_reference.startswith("edge_id:"):
                normalized_reference = f"edge:{raw_reference[len('edge_id:'):]}"
            if normalized_reference not in provenance:
                provenance.append(normalized_reference)
        return provenance

    def _fallback_provenance(self, *, evidence: EvidenceBundle) -> list[str]:
        provenance = [f"edge:{edge.edge_id}" for edge in evidence.subgraph_edges[:2]]
        if len(provenance) < 2:
            provenance.extend(f"PMID:{passage.pmid}" for passage in evidence.pubmed_passages[: 2 - len(provenance)])
        return provenance[:2]

    def _build_trace(
        self,
        *,
        state: PipelineState | None,
        trm_output: TRMOutput,
        provenance: list[str],
    ) -> list[dict[str, Any]]:
        trace = list(trm_output.trace)
        orchestration_event = {
            "stage": "orchestrator",
            "pipeline_version": DEFAULT_PIPELINE_TRACE_VERSION,
            "question": state.question if state is not None else None,
            "provenance": provenance,
            "events": list(state.events) if state is not None else [],
            "errors": list(state.errors) if state is not None else [],
        }
        trace.append(orchestration_event)
        return trace

    def _record_event(self, state: PipelineState, stage: str, payload: dict[str, Any]) -> None:
        state.events.append({"stage": stage, **payload})

    def _deduplicate_judge_outputs(self, judge_outputs: list[JudgeOutput]) -> list[JudgeOutput]:
        deduplicated: list[JudgeOutput] = []
        seen_payloads: set[str] = set()
        for judge_output in judge_outputs:
            payload = judge_output.model_dump_json()
            if payload in seen_payloads:
                continue
            seen_payloads.add(payload)
            deduplicated.append(judge_output)
        return deduplicated

    def _emergency_abstain(self, *, question: str, reason: str, state: PipelineState | None = None) -> AnswerWithTrace:
        question_id = str(uuid4())
        fallback_provenance = ["edge:emergency-abstain"]
        answer_text = (
            f"ABSTENTION: I cannot safely answer the question because the pipeline failed before verified evidence was assembled [edge:emergency-abstain].\n"
            f"Reasoning: The orchestrator captured the failure reason as {reason.strip() or 'unknown error'} [edge:emergency-abstain].\n"
            "Confidence level: low (0.00) [edge:emergency-abstain]."
        )
        if state is not None:
            state.answer_text = answer_text
            state.errors.append(reason)
        return AnswerWithTrace(
            answer_text=answer_text,
            question_id=question_id,
            provenance=fallback_provenance,
            confidence=0.0,
            abstention=True,
            abstention_reason=reason,
            judge_outputs=[],
            trm_trace=[{
                "stage": "orchestrator_emergency_abstain",
                "pipeline_version": DEFAULT_PIPELINE_TRACE_VERSION,
                "question": question,
                "reason": reason,
            }],
        )

    @staticmethod
    def _humanize_identifier(identifier: str) -> str:
        normalized = identifier.split(":", 1)[-1]
        return normalized.replace("_", " ").replace("-", " ").strip() or identifier

    @staticmethod
    def _is_timeout_error(exc: BaseException) -> bool:
        return isinstance(exc, (TimeoutError, requests.Timeout)) or "timeout" in str(exc).lower()

    @staticmethod
    def _is_oom_error(exc: BaseException) -> bool:
        message = str(exc).lower()
        return "out of memory" in message or "cuda oom" in message or "cuda out of memory" in message


@dataclass(slots=True)
class PipelineAbstention(Exception):
    reason: str
    judge_outputs: list[JudgeOutput] = field(default_factory=list)


__all__ = [
    "MedicalReasoningSystemV3",
    "PipelineAbstention",
    "PipelineState",
    "SystemConfig",
]
