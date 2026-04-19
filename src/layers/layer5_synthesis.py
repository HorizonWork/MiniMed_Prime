from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

from src.schemas import EvidenceBundle, JudgeOutput, KGEdge, TRMOutput
from src.utils.kaggle_env import KaggleEnv
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


DEFAULT_SYNTHESIS_SYSTEM_PROMPT = """You are a medical expert synthesizing an answer from verified evidence.
RULES:
1. Use ONLY provided evidence paths and PubMed passages.
2. Tag EVERY sentence: [edge:EDGE_ID] for KG facts, [PMID:XXXXX] for literature.
3. If confidence < 0.7, use uncertainty phrasing ("Evidence suggests...").
4. If confidence < 0.4 or contradiction detected, output ABSTENTION message.
5. NEVER generate a PMID or edge_id not in the provided evidence.
6. For drug claims, include therapeutic dosage range if available.
7. Format: Direct answer -> Reasoning with citations -> Confidence level."""

DEFAULT_SYNTHESIS_TEMPERATURE = 0.3
DEFAULT_SYNTHESIS_MAX_TOKENS = 1024
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.4
DEFAULT_UNCERTAINTY_THRESHOLD = 0.7
DEFAULT_MAX_PASSAGES = 5
DEFAULT_MAX_PATHS = 5
DEFAULT_MAX_EVIDENCE_SENTENCES = 3
DEFAULT_PROVENANCE_SENTENCE_SPLIT = r"(?<=[.!?])\s+|\n+"
DEFAULT_TAG_PATTERN = re.compile(r"\[(edge:[^\]]+|edge_id:[^\]]+|PMID:[^\]]+)\]")
DEFAULT_DOSAGE_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:mcg|mg|g|mL|units)(?:\s?(?:/|per)\s?(?:day|dose|kg|week))?(?:\s?(?:to|-)\s?\d+(?:\.\d+)?\s?(?:mcg|mg|g|mL|units)(?:\s?(?:/|per)\s?(?:day|dose|kg|week))?)?",
    re.IGNORECASE,
)
DEFAULT_NEGATIVE_RELATIONS = {"contraindication", "disease_phenotype_negative", "anatomy_protein_absent"}
DEFAULT_OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
SUPPORTED_SYNTHESIS_BACKENDS = {"auto", "gguf", "transformers_4bit", "openai"}


class AnswerSynthesizer:
    def __init__(self, model_path: str, backend: str = "auto", n_ctx: int = 8192) -> None:
        self.model_path = _resolve_local_model_path(model_path)
        self.requested_backend = backend
        self.n_ctx = n_ctx
        self.system_prompt = DEFAULT_SYNTHESIS_SYSTEM_PROMPT
        self.backend = "heuristic"
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.last_clean_answer = ""
        self.last_provenance_issues: list[str] = []
        self.last_backend_used = "heuristic"
        self.requests_session = requests.Session()
        self.event_logger = StructuredLogger("layer5_synthesis", DEFAULT_LOG_DIR)
        self._load_backend()

    def synthesize(
        self,
        question: str,
        trm_output: TRMOutput,
        evidence: EvidenceBundle,
        judge_feedback: JudgeOutput | dict[str, Any] | None = None,
    ) -> str:
        if isinstance(judge_feedback, dict):
            judge_feedback = JudgeOutput.model_validate(judge_feedback)
        self.last_backend_used = self.backend
        self.event_logger.log_event(
            "synthesis_start",
            {
                "backend": self.backend,
                "temperature": DEFAULT_SYNTHESIS_TEMPERATURE,
                "backend_used": self.backend,
                "latency_ms": 0.0,
            },
        )

        if self._should_abstain(trm_output=trm_output, judge_feedback=judge_feedback):
            abstention = self.format_abstention(
                reason=self._abstention_reason(trm_output=trm_output, judge_feedback=judge_feedback),
                evidence_stats=self._build_evidence_stats(trm_output=trm_output, evidence=evidence),
            )
            cleaned_answer, _, issues = self._sanitize_answer(abstention, evidence)
            self.last_clean_answer = cleaned_answer
            self.last_provenance_issues = issues
            self.event_logger.log_event(
                "answer_finalized",
                {
                    "abstention": True,
                    "confidence": trm_output.validity_score,
                    "char_length": len((cleaned_answer or abstention)),
                    "backend_used": self.backend,
                    "latency_ms": 0.0,
                },
            )
            return cleaned_answer or abstention

        prompt = self._build_prompt(
            question=question,
            trm_output=trm_output,
            evidence=evidence,
            judge_feedback=judge_feedback,
        )
        candidate_answer: str | None = None
        if self.backend == "llama_cpp":
            candidate_answer = self._generate_with_llama_cpp(prompt)
        elif self.backend == "transformers_4bit":
            candidate_answer = self._generate_with_transformers(prompt)
        elif self.backend == "openai":
            candidate_answer = self._generate_with_openai(prompt)

        if candidate_answer:
            cleaned_answer, is_clean, issues = self._sanitize_answer(candidate_answer, evidence)
            if cleaned_answer and is_clean:
                self.last_clean_answer = cleaned_answer
                self.last_provenance_issues = issues
                self.event_logger.log_event(
                    "answer_finalized",
                    {
                        "abstention": False,
                        "confidence": trm_output.validity_score,
                        "char_length": len(cleaned_answer),
                        "backend_used": self.backend,
                        "latency_ms": 0.0,
                    },
                )
                return cleaned_answer
            logger.debug("Generated synthesis failed provenance enforcement; falling back to deterministic synthesis.")

        heuristic_answer = self._build_heuristic_answer(
            question=question,
            trm_output=trm_output,
            evidence=evidence,
            judge_feedback=judge_feedback,
        )
        cleaned_answer, _, issues = self._sanitize_answer(heuristic_answer, evidence)
        self.last_clean_answer = cleaned_answer
        self.last_provenance_issues = issues
        if cleaned_answer:
            self.event_logger.log_event(
                "answer_finalized",
                {
                    "abstention": False,
                    "confidence": trm_output.validity_score,
                    "char_length": len(cleaned_answer),
                    "backend_used": self.backend,
                    "latency_ms": 0.0,
                },
            )
            return cleaned_answer

        abstention = self.format_abstention(
            reason="Provenance enforcement removed all generated content.",
            evidence_stats=self._build_evidence_stats(trm_output=trm_output, evidence=evidence),
        )
        cleaned_abstention, _, abstention_issues = self._sanitize_answer(abstention, evidence)
        self.last_clean_answer = cleaned_abstention
        self.last_provenance_issues = [*issues, *abstention_issues]
        self.event_logger.log_event(
            "answer_finalized",
            {
                "abstention": True,
                "confidence": trm_output.validity_score,
                "char_length": len((cleaned_abstention or abstention)),
                "backend_used": self.backend,
                "latency_ms": 0.0,
            },
        )
        return cleaned_abstention or abstention

    def enforce_provenance(self, answer: str, evidence: EvidenceBundle) -> tuple[bool, list[str]]:
        cleaned_answer, is_clean, issues = self._sanitize_answer(answer, evidence)
        self.last_clean_answer = cleaned_answer
        self.last_provenance_issues = issues
        return is_clean, issues

    def format_abstention(self, reason: str, evidence_stats: dict[str, Any]) -> str:
        provenance_tags = evidence_stats.get("provenance_tags") or ["[edge:unknown]"]
        primary_tags = " ".join(provenance_tags[:2])
        score = float(evidence_stats.get("validity_score", 0.0))
        edge_count = int(evidence_stats.get("edge_count", 0))
        passage_count = int(evidence_stats.get("passage_count", 0))
        normalized_reason = reason.strip().rstrip(".!?").lower() or "the evidence is insufficient"

        sentences = [
            f"ABSTENTION: I cannot provide a confident medical answer because {normalized_reason} {primary_tags}.",
            f"Reasoning: The verified evidence bundle contains {edge_count} KG edges and {passage_count} PubMed passages, but the reasoning signal remained insufficient or contradictory {primary_tags}.",
            f"Confidence level: low ({score:.2f}) {primary_tags}.",
        ]
        return "\n".join(sentences)

    def _load_backend(self) -> None:
        if self.model_path.lower() == "heuristic":
            self.backend = "heuristic"
            self.last_backend_used = self.backend
            return

        requested_backend = self.requested_backend.lower().strip()
        if requested_backend not in SUPPORTED_SYNTHESIS_BACKENDS:
            logger.warning(
                "Unsupported synthesis backend {} requested for {}; falling back to heuristic.",
                self.requested_backend,
                self.model_path,
            )
            self.backend = "heuristic"
            self.last_backend_used = self.backend
            return

        selected_backend = self._select_backend(requested_backend)
        loaded = False
        if selected_backend == "openai":
            loaded = self._load_openai_backend()
        elif selected_backend == "gguf":
            loaded = self._load_gguf_backend()
            if not loaded and requested_backend == "auto":
                loaded = self._load_transformers_backend()
        elif selected_backend == "transformers_4bit":
            loaded = self._load_transformers_backend()
            if not loaded and requested_backend == "auto" and self._resolve_gguf_file() is not None:
                loaded = self._load_gguf_backend()

        if not loaded:
            self.backend = "heuristic"
            self.last_backend_used = self.backend

    def _select_backend(self, requested_backend: str) -> str:
        if requested_backend != "auto":
            return requested_backend
        if self._should_use_openai_model():
            return "openai"
        return "gguf" if self._resolve_gguf_file() is not None else "transformers_4bit"

    def _should_use_openai_model(self) -> bool:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return False
        candidate = Path(self.model_path)
        if candidate.exists():
            return False
        return _looks_like_openai_model_name(self.model_path)

    def _load_openai_backend(self) -> bool:
        if not self._should_use_openai_model():
            return False
        self.backend = "openai"
        self.last_backend_used = self.backend
        self.event_logger.log_event(
            "synthesis_backend_loaded",
            {
                "backend": self.backend,
                "model_path": self.model_path,
                "parameter_count": None,
                "cuda_memory_gb": self._cuda_memory_stats_gb(),
                "backend_used": self.backend,
                "latency_ms": 0.0,
            },
        )
        return True

    def _load_gguf_backend(self) -> bool:
        model_path = self._resolve_gguf_file()
        if model_path is None:
            return False

        try:  # pragma: no cover - optional dependency path
            from llama_cpp import Llama

            n_gpu_layers = -1 if torch is not None and torch.cuda.is_available() else 0
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=n_gpu_layers,
                main_gpu=0,
                verbose=False,
            )
            self.tokenizer = None
            self.backend = "llama_cpp"
            self.last_backend_used = self.backend
            return True
        except Exception as exc:
            logger.warning("Failed to load llama.cpp synthesis backend from {}: {}", model_path, exc)
            return False

    def _load_transformers_backend(self) -> bool:
        if torch is None:
            return False

        try:  # pragma: no cover - optional dependency path
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.backend = "transformers_4bit"
            self.last_backend_used = self.backend
            self.event_logger.log_event(
                "synthesis_backend_loaded",
                {
                    "backend": self.backend,
                    "model_path": self.model_path,
                    "parameter_count": self._safe_parameter_count(),
                    "cuda_memory_gb": self._cuda_memory_stats_gb(),
                    "backend_used": self.backend,
                    "latency_ms": 0.0,
                },
            )
            return True
        except Exception as exc:
            logger.warning("Failed to load transformers 4-bit synthesis backend from {}: {}", self.model_path, exc)
            self.model = None
            self.tokenizer = None
            return False

    def _resolve_gguf_file(self) -> Path | None:
        candidate = Path(self.model_path)
        if candidate.is_file() and candidate.suffix.lower() == ".gguf":
            return candidate
        if candidate.is_dir():
            gguf_candidates = sorted(path for path in candidate.iterdir() if path.is_file() and path.suffix.lower() == ".gguf")
            if gguf_candidates:
                preferred_candidates = [
                    path for path in gguf_candidates if "q4_k_m" in path.name.lower() or "q4-k-m" in path.name.lower()
                ]
                return preferred_candidates[0] if preferred_candidates else gguf_candidates[0]
        try:
            if any(name.lower().endswith(".gguf") for name in os.listdir(self.model_path)):
                candidate_dir = Path(self.model_path)
                if candidate_dir.exists():
                    return self._resolve_gguf_file_from_dir(candidate_dir)
        except OSError:
            return None
        return None

    def _resolve_gguf_file_from_dir(self, directory: Path) -> Path | None:
        gguf_candidates = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".gguf")
        if not gguf_candidates:
            return None
        preferred_candidates = [
            path for path in gguf_candidates if "q4_k_m" in path.name.lower() or "q4-k-m" in path.name.lower()
        ]
        return preferred_candidates[0] if preferred_candidates else gguf_candidates[0]

    def _build_prompt(
        self,
        *,
        question: str,
        trm_output: TRMOutput,
        evidence: EvidenceBundle,
        judge_feedback: JudgeOutput | None,
    ) -> str:
        allowed_refs = self._allowed_reference_ids(evidence)
        path_context = [
            {
                "nodes": path.nodes,
                "edges": path.edges,
                "confidence": path.confidence,
                "supporting_pmids": path.supporting_pmids,
            }
            for path in trm_output.ranked_paths[:DEFAULT_MAX_PATHS]
        ]
        passage_context = [
            {
                "pmid": passage.pmid,
                "title": passage.title,
                "abstract": passage.abstract,
            }
            for passage in evidence.pubmed_passages[:DEFAULT_MAX_PASSAGES]
        ]
        edge_context = [
            {
                "edge_id": edge.edge_id,
                "statement": self._edge_statement(edge),
                "reliability": edge.source_reliability * edge.amg_confidence,
            }
            for edge in evidence.subgraph_edges
        ]
        dosage_context = self._extract_dosage_mentions(evidence)
        judge_context = judge_feedback.model_dump() if judge_feedback is not None else None
        return (
            f"{self.system_prompt}\n\n"
            f"Question: {question}\n"
            f"TRM validity score: {trm_output.validity_score:.2f}\n"
            f"TRM contradiction flag: {trm_output.contradiction_flag}\n"
            f"Allowed provenance tags: {', '.join(sorted(allowed_refs))}\n"
            f"Reasoning paths: {path_context}\n"
            f"KG evidence: {edge_context}\n"
            f"PubMed evidence: {passage_context}\n"
            f"Dosage mentions: {dosage_context}\n"
            f"Judge feedback: {judge_context}\n"
            "Draft the final answer now."
        )

    def _generate_with_llama_cpp(self, prompt: str) -> str:
        completion = self.model.create_completion(  # type: ignore[union-attr]  # pragma: no cover - optional path
            prompt=prompt,
            max_tokens=DEFAULT_SYNTHESIS_MAX_TOKENS,
            temperature=DEFAULT_SYNTHESIS_TEMPERATURE,
            stop=["\n\nQuestion:", "\n\nUser:"],
        )
        choices = completion.get("choices", [])
        if not choices:
            return ""
        return str(choices[0].get("text", "")).strip()

    def _generate_with_transformers(self, prompt: str) -> str:
        if self.model is None or self.tokenizer is None or torch is None:
            return ""

        max_prompt_tokens = max(self.n_ctx - DEFAULT_SYNTHESIS_MAX_TOKENS, 256)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens,
        )
        model_device = getattr(self.model, "device", None)
        if model_device is not None and getattr(model_device, "type", None) != "meta":
            encoded = {
                key: value.to(model_device) if hasattr(value, "to") else value for key, value in encoded.items()
            }
        max_new_tokens = DEFAULT_SYNTHESIS_MAX_TOKENS
        attempt_index = 1
        while True:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            started_at = time.perf_counter()
            try:
                with torch.no_grad():
                    generated = self.model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        temperature=DEFAULT_SYNTHESIS_TEMPERATURE,
                        top_p=0.9,
                        do_sample=DEFAULT_SYNTHESIS_TEMPERATURE > 0,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                prompt_length = encoded["input_ids"].shape[-1]
                generated_tokens = generated[0][prompt_length:]
                self.event_logger.log_event(
                    "transformers_generation",
                    {
                        "attempt": attempt_index,
                        "prompt_tokens": int(prompt_length),
                        "generated_tokens": int(generated_tokens.shape[-1]),
                        "max_new_tokens": int(max_new_tokens),
                        "temperature": DEFAULT_SYNTHESIS_TEMPERATURE,
                        "top_p": 0.9,
                        "cuda_memory_gb": self._cuda_memory_stats_gb(),
                        "backend_used": self.backend,
                        "latency_ms": (time.perf_counter() - started_at) * 1000.0,
                    },
                )
                return str(self.tokenizer.decode(generated_tokens, skip_special_tokens=True)).strip()
            except RuntimeError as exc:
                if self._is_cuda_oom(exc) and attempt_index == 1:
                    retry_max_new_tokens = max(max_new_tokens // 2, 64)
                    self.event_logger.log_exception(
                        "transformers_generation_oom_retry",
                        exc,
                        {
                            "attempt": attempt_index,
                            "max_new_tokens": max_new_tokens,
                            "retry_max_new_tokens": retry_max_new_tokens,
                            "cuda_memory_gb": self._cuda_memory_stats_gb(),
                            "backend_used": self.backend,
                            "latency_ms": (time.perf_counter() - started_at) * 1000.0,
                        },
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    max_new_tokens = retry_max_new_tokens
                    attempt_index += 1
                    continue
                self.event_logger.log_exception(
                    "transformers_generation_failed",
                    exc,
                    {
                        "attempt": attempt_index,
                        "max_new_tokens": max_new_tokens,
                        "cuda_memory_gb": self._cuda_memory_stats_gb(),
                        "backend_used": self.backend,
                        "latency_ms": (time.perf_counter() - started_at) * 1000.0,
                    },
                )
                return ""

    def _generate_with_openai(self, prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return ""
        started_at = time.perf_counter()
        try:
            response = self.requests_session.post(
                _resolve_openai_chat_completions_url(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                json={
                    "model": self.model_path,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": DEFAULT_SYNTHESIS_TEMPERATURE,
                    "max_tokens": DEFAULT_SYNTHESIS_MAX_TOKENS,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return ""
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(str(part.get("text", "")))
                    elif isinstance(part, dict) and "text" in part:
                        text_parts.append(str(part["text"]))
                content = "\n".join(part for part in text_parts if part)
            self.event_logger.log_event(
                "openai_generation",
                {
                    "prompt_chars": len(prompt),
                    "temperature": DEFAULT_SYNTHESIS_TEMPERATURE,
                    "max_tokens": DEFAULT_SYNTHESIS_MAX_TOKENS,
                    "cuda_memory_gb": self._cuda_memory_stats_gb(),
                    "backend_used": self.backend,
                    "latency_ms": (time.perf_counter() - started_at) * 1000.0,
                },
            )
            return str(content).strip()
        except Exception as exc:
            self.event_logger.log_exception(
                "openai_generation_failed",
                exc,
                {
                    "prompt_chars": len(prompt),
                    "max_tokens": DEFAULT_SYNTHESIS_MAX_TOKENS,
                    "cuda_memory_gb": self._cuda_memory_stats_gb(),
                    "backend_used": self.backend,
                    "latency_ms": (time.perf_counter() - started_at) * 1000.0,
                },
            )
            return ""

    def _safe_parameter_count(self) -> int | None:
        if self.model is None or not hasattr(self.model, "parameters"):
            return None
        try:
            return int(sum(parameter.numel() for parameter in self.model.parameters()))
        except Exception:
            return None

    @staticmethod
    def _cuda_memory_stats_gb() -> dict[str, float]:
        if torch is None or not torch.cuda.is_available():
            return {"allocated": 0.0, "reserved": 0.0}
        return {
            "allocated": float(torch.cuda.memory_allocated() / 1e9),
            "reserved": float(torch.cuda.memory_reserved() / 1e9),
        }

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        message = str(exc).lower()
        return "out of memory" in message or "cuda oom" in message or "cuda out of memory" in message

    def _build_heuristic_answer(
        self,
        *,
        question: str,
        trm_output: TRMOutput,
        evidence: EvidenceBundle,
        judge_feedback: JudgeOutput | None,
    ) -> str:
        del question
        del judge_feedback
        top_path = trm_output.ranked_paths[0] if trm_output.ranked_paths else None
        direct_tags = self._format_tags(self._select_provenance_tags(evidence=evidence, path=top_path, limit=3))
        reasoning_tags = self._format_tags(self._select_reasoning_tags(evidence=evidence, path=top_path))
        confidence_tags = self._format_tags(self._select_provenance_tags(evidence=evidence, path=top_path, limit=2))

        if trm_output.validity_score < DEFAULT_UNCERTAINTY_THRESHOLD:
            direct_prefix = "Direct answer: Evidence suggests"
        else:
            direct_prefix = "Direct answer: The evidence supports"

        direct_statement = self._direct_claim_text(top_path=top_path, evidence=evidence)
        reasoning_statement = self._reasoning_text(top_path=top_path, evidence=evidence)
        dosage_statement = self._dosage_text(evidence)
        confidence_label = self._confidence_label(trm_output.validity_score)

        sentences = [
            f"{direct_prefix} {direct_statement} {direct_tags}.",
            f"Reasoning: {reasoning_statement} {reasoning_tags}.",
        ]
        if dosage_statement is not None:
            sentences.append(f"Reasoning: {dosage_statement} {reasoning_tags}.")
        sentences.append(
            f"Confidence level: {confidence_label} ({trm_output.validity_score:.2f}) {confidence_tags}."
        )
        return "\n".join(sentences)

    def _sanitize_answer(self, answer: str, evidence: EvidenceBundle) -> tuple[str, bool, list[str]]:
        allowed_refs = self._allowed_reference_ids(evidence)
        cleaned_sentences: list[str] = []
        issues: list[str] = []
        for sentence in self._split_sentences(answer):
            normalized_sentence, sentence_issues = self._sanitize_sentence(sentence, allowed_refs)
            if sentence_issues:
                issues.extend(sentence_issues)
            if normalized_sentence:
                cleaned_sentences.append(normalized_sentence)

        cleaned_answer = "\n".join(cleaned_sentences)
        is_clean = not issues
        self.event_logger.log_event(
            "provenance_enforced",
            {
                "sentences_total": len(self._split_sentences(answer)),
                "sentences_dropped": max(len(self._split_sentences(answer)) - len(cleaned_sentences), 0),
                "issues": issues,
                "backend_used": self.backend,
                "latency_ms": 0.0,
            },
        )
        return cleaned_answer, is_clean, issues

    def _sanitize_sentence(self, sentence: str, allowed_refs: set[str]) -> tuple[str | None, list[str]]:
        normalized_sentence = " ".join(sentence.strip().split())
        if not normalized_sentence:
            return None, []

        tags = DEFAULT_TAG_PATTERN.findall(normalized_sentence)
        if not tags:
            return None, [f"Dropped sentence without provenance: {normalized_sentence}"]

        normalized_tags = [self._normalize_reference(tag) for tag in tags]
        invalid_tags = [tag for tag in normalized_tags if tag not in allowed_refs]
        if invalid_tags:
            return None, [f"Dropped sentence with invalid provenance {invalid_tags}: {normalized_sentence}"]

        for raw_tag, normalized_tag in zip(tags, normalized_tags):
            if raw_tag != normalized_tag:
                normalized_sentence = normalized_sentence.replace(f"[{raw_tag}]", f"[{normalized_tag}]")

        if normalized_sentence[-1] not in ".!?":
            normalized_sentence = f"{normalized_sentence}."
        return normalized_sentence, []

    def _should_abstain(self, trm_output: TRMOutput, judge_feedback: JudgeOutput | None) -> bool:
        if trm_output.validity_score < DEFAULT_LOW_CONFIDENCE_THRESHOLD:
            return True
        if trm_output.contradiction_flag:
            return True
        if judge_feedback is None:
            return False
        if judge_feedback.h5_present:
            return True
        if judge_feedback.overall_recommendation == "ABSTAIN":
            return True
        if judge_feedback.faithfulness_score < DEFAULT_LOW_CONFIDENCE_THRESHOLD:
            return True
        return False

    def _abstention_reason(self, trm_output: TRMOutput, judge_feedback: JudgeOutput | None) -> str:
        if judge_feedback is not None and judge_feedback.abstention_reason:
            return judge_feedback.abstention_reason
        if trm_output.contradiction_flag:
            return "the evidence is contradictory"
        if trm_output.validity_score < DEFAULT_LOW_CONFIDENCE_THRESHOLD:
            return "the verified reasoning confidence is below the abstention threshold"
        return "the verified evidence is insufficient"

    def _build_evidence_stats(self, trm_output: TRMOutput, evidence: EvidenceBundle) -> dict[str, Any]:
        return {
            "validity_score": trm_output.validity_score,
            "edge_count": len(evidence.subgraph_edges),
            "passage_count": len(evidence.pubmed_passages),
            "provenance_tags": [f"[{tag}]" for tag in self._select_provenance_tags(evidence=evidence, path=None, limit=2)],
        }

    def _allowed_reference_ids(self, evidence: EvidenceBundle) -> set[str]:
        edge_refs = {f"edge:{edge.edge_id}" for edge in evidence.subgraph_edges}
        pmid_refs = {f"PMID:{passage.pmid}" for passage in evidence.pubmed_passages}
        return edge_refs | pmid_refs

    def _select_provenance_tags(
        self,
        *,
        evidence: EvidenceBundle,
        path: Any | None,
        limit: int,
    ) -> list[str]:
        selected: list[str] = []
        if path is not None:
            for edge_id in getattr(path, "edges", []):
                tag = f"edge:{edge_id}"
                if tag not in selected:
                    selected.append(tag)
                if len(selected) >= limit:
                    return selected
            for pmid in getattr(path, "supporting_pmids", []):
                tag = f"PMID:{pmid}"
                if tag not in selected:
                    selected.append(tag)
                if len(selected) >= limit:
                    return selected

        for edge in evidence.subgraph_edges:
            tag = f"edge:{edge.edge_id}"
            if tag not in selected:
                selected.append(tag)
            if len(selected) >= limit:
                return selected

        for passage in evidence.pubmed_passages:
            tag = f"PMID:{passage.pmid}"
            if tag not in selected:
                selected.append(tag)
            if len(selected) >= limit:
                return selected
        return selected

    def _select_reasoning_tags(self, *, evidence: EvidenceBundle, path: Any | None) -> list[str]:
        tags = self._select_provenance_tags(evidence=evidence, path=path, limit=4)
        if len(tags) < 2:
            fallback = self._select_provenance_tags(evidence=evidence, path=None, limit=4)
            for tag in fallback:
                if tag not in tags:
                    tags.append(tag)
        return tags[:4]

    @staticmethod
    def _format_tags(tags: list[str]) -> str:
        return " ".join(f"[{tag}]" for tag in tags)

    def _direct_claim_text(self, *, top_path: Any | None, evidence: EvidenceBundle) -> str:
        if top_path is None or not top_path.edges:
            if evidence.pubmed_passages:
                return f"the literature provides only limited support relevant to {evidence.question_text.lower()}"
            return "the available evidence does not support a specific conclusion"

        edge_lookup = {edge.edge_id: edge for edge in evidence.subgraph_edges}
        first_edge = edge_lookup.get(top_path.edges[0])
        if first_edge is None:
            return "the top reasoning path connects the queried entities"

        head_text = _humanize_identifier(first_edge.head)
        tail_text = _humanize_identifier(first_edge.tail)
        if first_edge.relation in DEFAULT_NEGATIVE_RELATIONS or "contra" in first_edge.display_relation.lower():
            return f"{head_text} is linked to a clinically important negative association with {tail_text}"
        if first_edge.relation == "drug_drug":
            return f"{head_text} has a clinically relevant interaction with {tail_text}"
        if first_edge.relation == "indication":
            return f"{head_text} is indicated for {tail_text}"
        return f"{head_text} {first_edge.display_relation.lower()} {tail_text}"

    def _reasoning_text(self, *, top_path: Any | None, evidence: EvidenceBundle) -> str:
        if top_path is None:
            if evidence.pubmed_passages:
                passage = evidence.pubmed_passages[0]
                return f"the most relevant literature comes from PMID {passage.pmid} and remains limited to the retrieved abstract"
            return "no connected reasoning path was available from the evidence bundle"

        edge_lookup = {edge.edge_id: edge for edge in evidence.subgraph_edges}
        statements: list[str] = []
        for edge_id in top_path.edges[:DEFAULT_MAX_EVIDENCE_SENTENCES]:
            edge = edge_lookup.get(edge_id)
            if edge is None:
                continue
            statements.append(self._edge_statement(edge))
        if not statements:
            return "the highest-ranked path did not decode to named PrimeKG relations"
        return "; then ".join(statements)

    def _dosage_text(self, evidence: EvidenceBundle) -> str | None:
        dosage_mentions = self._extract_dosage_mentions(evidence)
        if not dosage_mentions:
            return None
        best_dosage = dosage_mentions[0]
        return f"a dosage mention available in the evidence is {best_dosage['mention']} from PMID {best_dosage['pmid']}"

    def _extract_dosage_mentions(self, evidence: EvidenceBundle) -> list[dict[str, str]]:
        dosage_mentions: list[dict[str, str]] = []
        for passage in evidence.pubmed_passages:
            text = f"{passage.title} {passage.abstract}"
            for match in DEFAULT_DOSAGE_PATTERN.finditer(text):
                dosage_mentions.append({"mention": match.group(0), "pmid": passage.pmid})
        return dosage_mentions

    @staticmethod
    def _confidence_label(score: float) -> str:
        if score < DEFAULT_LOW_CONFIDENCE_THRESHOLD:
            return "low"
        if score < DEFAULT_UNCERTAINTY_THRESHOLD:
            return "moderate"
        return "high"

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        fragments = re.split(DEFAULT_PROVENANCE_SENTENCE_SPLIT, text.strip())
        return [fragment.strip() for fragment in fragments if fragment and fragment.strip()]

    @staticmethod
    def _normalize_reference(reference: str) -> str:
        if reference.startswith("edge_id:"):
            return f"edge:{reference[len('edge_id:'):]}"
        return reference

    @staticmethod
    def _edge_statement(edge: KGEdge) -> str:
        return f"{_humanize_identifier(edge.head)} {edge.display_relation.lower()} {_humanize_identifier(edge.tail)}"


def _humanize_identifier(identifier: str) -> str:
    normalized = identifier.split(":", 1)[-1]
    return normalized.replace("_", " ").replace("-", " ").strip() or identifier


__all__ = ["AnswerSynthesizer"]


def _resolve_local_model_path(model_path: str) -> str:
    candidate = Path(model_path)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    if str(model_path).replace("\\", "/").startswith("data/"):
        resolved = KaggleEnv.path(model_path)
        if resolved.exists():
            return str(resolved)
    return model_path


def _looks_like_openai_model_name(model_name: str) -> bool:
    normalized = model_name.strip().lower()
    return normalized.startswith(("gpt-", "o1", "o3", "o4")) or normalized in {
        "4omini",
        "4o-mini",
        "gpt4omini",
        "gpt-4omini",
    }


def _resolve_openai_chat_completions_url() -> str:
    configured = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_API_URL).rstrip("/")
    if configured.endswith("/chat/completions"):
        return configured
    if configured.endswith("/v1"):
        return f"{configured}/chat/completions"
    return f"{configured}/chat/completions"
