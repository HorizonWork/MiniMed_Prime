from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Type, TypeVar

import requests
from pydantic import BaseModel, ConfigDict, Field

from src.schemas import (
    ClaimVerdict,
    DEFAULT_ABSTENTION_REASON_H5,
    EvidenceBundle,
    JudgeOutput,
    QuestionEntity,
    Recommendation,
    TRMOutput,
)
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


SchemaT = TypeVar("SchemaT", bound=BaseModel)

DEFAULT_JUDGE_MODEL_NAME = "medical_o1_verifier_3B"
DEFAULT_OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_REMOTE_TIMEOUT_SECONDS = 30
DEFAULT_MAX_LLM_RETRIES = 3
DEFAULT_REVISION_RETRIES = 2
DEFAULT_JSON_SCHEMA_INSTRUCTION = (
    "Return strict JSON only. Do not include markdown, commentary, or code fences."
)
DEFAULT_LOCAL_GENERATION_TOKENS = 512
DEFAULT_NLI_MODEL_NAME = "roberta-large-mnli"
DEFAULT_LIFE_CRITICAL_TERMS = {
    "contraindicated",
    "contraindication",
    "unsafe",
    "fatal",
    "life-threatening",
    "bleeding",
    "hemorrhage",
    "pregnancy",
    "pregnant",
    "child",
    "children",
    "pediatric",
    "dose",
    "dosage",
    "mg",
    "mcg",
    "g/day",
    "avoid",
    "warning",
    "toxicity",
    "overdose",
    "renal failure",
    "hepatic failure",
}
DEFAULT_TEMPORAL_TERMS = {"acute", "chronic", "today", "tomorrow", "week", "month", "year", "duration"}
DEFAULT_POPULATION_TERMS = {
    "adult",
    "adults",
    "child",
    "children",
    "pediatric",
    "elderly",
    "geriatric",
    "pregnant",
    "pregnancy",
    "male",
    "female",
}
DEFAULT_LOGICAL_TERMS = {"because", "therefore", "thus", "hence", "so that", "mechanism", "pathway"}
DEFAULT_NEGATION_TERMS = {
    "no",
    "not",
    "never",
    "without",
    "contraindicated",
    "avoid",
    "unsafe",
    "worsen",
    "worsens",
    "worsened",
    "increase",
    "increases",
    "risk",
    "adverse",
    "harmful",
}
DEFAULT_POSITIVE_TERMS = {
    "safe",
    "effective",
    "recommended",
    "beneficial",
    "indicated",
    "improves",
    "reduces",
    "prevents",
    "treats",
    "helps",
}
DEFAULT_RELATION_NEGATIVE_TERMS = {
    "contraindication",
    "disease_phenotype_negative",
    "anatomy_protein_absent",
}
DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
DEFAULT_ENTITY_WEIGHT = 0.25
DEFAULT_GROUNDING_WEIGHT = 0.45
DEFAULT_SOUNDNESS_WEIGHT = 0.30
DEFAULT_ACCEPT_THRESHOLD = 0.75
DEFAULT_ABSTAIN_THRESHOLD = 0.40
DEFAULT_SUPPORT_THRESHOLD = 0.25
DEFAULT_CONTRADICTION_THRESHOLD = 0.30
DEFAULT_QUESTION_ENTITY_COVERAGE_THRESHOLD = 0.50
DEFAULT_EXPECTED_EDGE_COUNT_COMPLEX = 2
DEFAULT_EXPECTED_EDGE_COUNT_SIMPLE = 1

ENTITY_VALIDATOR_SYSTEM_PROMPT = (
    "You validate biomedical entity linking. Reject fake CUIs, wrong PrimeKG mappings, and unsupported expansions."
)
GROUNDING_INSPECTOR_SYSTEM_PROMPT = (
    "You verify that medical claims are grounded in supplied KG edges and PubMed evidence. "
    "Mark unsupported or contradicted claims conservatively."
)
SOUNDNESS_AUDITOR_SYSTEM_PROMPT = (
    "You audit reasoning chains for skipped steps, unsafe medical logic, and contradiction risk."
)
FAITHFULNESS_GUARDIAN_SYSTEM_PROMPT = (
    "You decide whether a medical answer is acceptable, requires revision, or must abstain."
)


class JudgeSchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EntityValidationResult(JudgeSchemaModel):
    valid: bool
    missing_entities: list[str] = Field(default_factory=list)
    mislinked: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ReasoningSoundnessResult(JudgeSchemaModel):
    sound: bool
    gaps: list[str] = Field(default_factory=list)
    rcs: float = Field(ge=0.0, le=1.0)
    rns: float = Field(ge=0.0, le=1.0)
    cdr: float = Field(ge=0.0, le=1.0)


class FaithfulnessGuardianResult(JudgeSchemaModel):
    verdict: Recommendation
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    h5_present: bool = False


class HallucinationJudgeResult(JudgeSchemaModel):
    answer_text: str
    final_recommendation: Recommendation
    confidence: float = Field(ge=0.0, le=1.0)
    retries_used: int = Field(ge=0, le=DEFAULT_REVISION_RETRIES)
    h5_present: bool
    abstention_reason: str | None = None
    entity_validation: EntityValidationResult
    grounding_output: JudgeOutput
    soundness_output: ReasoningSoundnessResult
    faithfulness_output: FaithfulnessGuardianResult
    revisions: list[str] = Field(default_factory=list)
    parse_success_rate: float = Field(ge=0.0, le=1.0)


class JudgeBase(ABC):
    def __init__(self, model_name: str, system_prompt: str, device: str = "cuda:1") -> None:
        self.model_name = _resolve_local_model_path(model_name)
        self.system_prompt = system_prompt
        self.device = device if device == "cpu" or self._cuda_available() else "cpu"
        self.backend = "heuristic"
        self.parse_attempts = 0
        self.parse_successes = 0
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.remote_endpoint: str | None = None
        self.requests_session = requests.Session()
        self.event_logger = StructuredLogger("layer4_judges", DEFAULT_LOG_DIR)
        self._load_backend()

    @abstractmethod
    def evaluate(self, **kwargs: Any) -> BaseModel:
        raise NotImplementedError

    @property
    def parse_success_rate(self) -> float:
        if self.parse_attempts == 0:
            return 1.0
        return self.parse_successes / self.parse_attempts

    def _call_llm(
        self,
        prompt: str,
        schema: Type[SchemaT],
        fallback_payload: dict[str, Any] | None = None,
    ) -> SchemaT:
        schema_instruction = json.dumps(schema.model_json_schema(), indent=2, sort_keys=True)
        constrained_prompt = (
            f"{self.system_prompt}\n\n"
            f"{DEFAULT_JSON_SCHEMA_INSTRUCTION}\n"
            f"Schema:\n{schema_instruction}\n\n"
            f"Task:\n{prompt}"
        )
        if self.backend == "heuristic":
            return self._validate_fallback(schema, fallback_payload)

        last_error: Exception | None = None
        for _ in range(DEFAULT_MAX_LLM_RETRIES):
            self.parse_attempts += 1
            try:
                raw_output = self._generate_structured_text(constrained_prompt)
                payload = self._extract_json_payload(raw_output)
                validated = schema.model_validate(payload)
                self.parse_successes += 1
                return validated
            except Exception as exc:  # pragma: no cover - exercised only with model backends available
                last_error = exc
                logger.warning("Judge structured generation failed for {}: {}", self.model_name, exc)
                self.event_logger.log_exception(
                    "judge_parse_failure",
                    exc,
                    {"judge_model": self.model_name, "backend_used": self.backend, "latency_ms": 0.0},
                )

        if fallback_payload is not None:
            logger.debug("Falling back to deterministic judge payload after parse failures.")
            return self._validate_fallback(schema, fallback_payload)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Judge generation failed without a fallback payload.")

    def _validate_fallback(self, schema: Type[SchemaT], fallback_payload: dict[str, Any] | None) -> SchemaT:
        if fallback_payload is None:
            raise RuntimeError("Fallback payload is required for heuristic execution.")
        self.parse_attempts += 1
        validated = schema.model_validate(fallback_payload)
        self.parse_successes += 1
        return validated

    def _load_backend(self) -> None:
        if self._should_use_openai():
            self.backend = "openai"
            self.remote_endpoint = _resolve_openai_chat_completions_url()
            return
        if self._should_use_gemini():
            self.backend = "gemini"
            self.remote_endpoint = DEFAULT_GEMINI_API_URL.format(model=self.model_name)
            return
        if self._try_load_llama_cpp():
            self.backend = "llama_cpp"
            return
        if self._try_load_transformers():
            self.backend = "transformers"
            return
        self.backend = "heuristic"

    def _should_use_openai(self) -> bool:
        api_key = os.getenv("OPENAI_API_KEY")
        normalized = self.model_name.lower().strip()
        if not api_key or not normalized:
            return False
        candidate_path = Path(self.model_name)
        if candidate_path.exists():
            return False
        return _looks_like_openai_model_name(normalized)

    def _should_use_gemini(self) -> bool:
        api_key = os.getenv("GEMINI_API_KEY")
        normalized = self.model_name.lower()
        return bool(api_key and "gemini" in normalized)

    def _try_load_llama_cpp(self) -> bool:
        model_path = Path(self.model_name)
        if model_path.suffix.lower() != ".gguf" or not model_path.exists():
            return False
        try:  # pragma: no cover - optional dependency path
            from llama_cpp import Llama

            self.model = Llama(model_path=str(model_path), n_ctx=4096, n_gpu_layers=-1 if self.device != "cpu" else 0)
            return True
        except Exception as exc:
            logger.warning("Failed to load llama.cpp judge model {}: {}", self.model_name, exc)
            return False

    def _try_load_transformers(self) -> bool:
        try:  # pragma: no cover - optional dependency path
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch_dtype = self._torch_dtype()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                local_files_only=True,
                torch_dtype=torch_dtype,
            )
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if hasattr(self.model, "to"):
                self.model.to(self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()
            return True
        except Exception as exc:
            logger.debug("No local transformers judge model available for {}: {}", self.model_name, exc)
            return False

    def _generate_structured_text(self, prompt: str) -> str:
        if self.backend == "openai":
            return self._generate_with_openai(prompt)
        if self.backend == "gemini":
            return self._generate_with_gemini(prompt)
        if self.backend == "llama_cpp":
            return self._generate_with_llama_cpp(prompt)
        if self.backend == "transformers":
            return self._generate_with_transformers(prompt)
        raise RuntimeError("Structured generation requested without an active LLM backend.")

    def _generate_with_openai(self, prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        endpoint = self.remote_endpoint or _resolve_openai_chat_completions_url()
        if not api_key:
            raise RuntimeError("OpenAI backend selected without OPENAI_API_KEY.")
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        response = self.requests_session.post(
            endpoint,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=DEFAULT_REMOTE_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("OpenAI response did not include choices.")
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
        if not str(content).strip():
            raise RuntimeError("OpenAI response did not include text content.")
        return str(content)

    def _generate_with_gemini(self, prompt: str) -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or self.remote_endpoint is None:  # pragma: no cover - guarded by backend selection
            raise RuntimeError("Gemini backend selected without GEMINI_API_KEY.")
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        }
        response = self.requests_session.post(
            self.remote_endpoint,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            json=payload,
            timeout=DEFAULT_REMOTE_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini response did not include candidates.")
        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
        output = "\n".join(text for text in texts if text)
        if not output:
            raise RuntimeError("Gemini response did not include text output.")
        return output

    def _generate_with_llama_cpp(self, prompt: str) -> str:
        completion = self.model.create_completion(  # type: ignore[union-attr]  # pragma: no cover - optional path
            prompt=prompt,
            max_tokens=DEFAULT_LOCAL_GENERATION_TOKENS,
            temperature=0,
            stop=["\n\n\n"],
        )
        choices = completion.get("choices", [])
        if not choices:
            raise RuntimeError("llama.cpp completion returned no choices.")
        return str(choices[0].get("text", ""))

    def _generate_with_transformers(self, prompt: str) -> str:
        if torch is None or self.model is None or self.tokenizer is None:  # pragma: no cover - guarded by loader
            raise RuntimeError("Transformers backend is not available.")
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=DEFAULT_LOCAL_GENERATION_TOKENS,
                do_sample=False,
                temperature=0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens[0], skip_special_tokens=True)

    @staticmethod
    def _extract_json_payload(raw_output: str) -> dict[str, Any]:
        stripped = raw_output.strip()
        if not stripped:
            raise ValueError("Judge model returned an empty response.")
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        start_index = stripped.find("{")
        end_index = stripped.rfind("}")
        if start_index == -1 or end_index == -1 or start_index >= end_index:
            raise ValueError("Judge output did not contain a JSON object.")
        payload = json.loads(stripped[start_index : end_index + 1])
        if not isinstance(payload, dict):
            raise ValueError("Judge output JSON must decode to an object.")
        return payload

    @staticmethod
    def _cuda_available() -> bool:
        return bool(torch is not None and torch.cuda.is_available())

    @staticmethod
    def _torch_dtype() -> Any:
        if torch is None:
            return None
        return torch.float16 if torch.cuda.is_available() else torch.float32

    def _log_invoked(self, judge_type: str) -> None:
        self.event_logger.log_event(
            "judge_invoked",
            {
                "judge_type": judge_type,
                "backend": self.backend,
                "backend_used": self.backend,
                "latency_ms": 0.0,
            },
        )

    def _log_verdict(self, judge_type: str, verdict: str, latency_ms: float) -> None:
        self.event_logger.log_event(
            "judge_verdict",
            {
                "judge_type": judge_type,
                "verdict": verdict,
                "backend": self.backend,
                "backend_used": self.backend,
                "latency_ms": latency_ms,
            },
        )


class EntityValidator(JudgeBase):
    def __init__(self, model_name: str = DEFAULT_JUDGE_MODEL_NAME, device: str = "cuda:1") -> None:
        super().__init__(model_name=model_name, system_prompt=ENTITY_VALIDATOR_SYSTEM_PROMPT, device=device)

    def evaluate(self, *, question_text: str, entities: list[QuestionEntity]) -> EntityValidationResult:
        self._log_invoked("entity")
        started_at = time.perf_counter()
        fallback = self._heuristic_validate(question_text=question_text, entities=entities)
        prompt = (
            f"Question: {question_text}\n"
            f"Entities: {json.dumps([entity.model_dump() for entity in entities], ensure_ascii=True)}\n"
            "Validate the biomedical entity links."
        )
        result = self._call_llm(prompt=prompt, schema=EntityValidationResult, fallback_payload=fallback.model_dump())
        self._log_verdict("entity", "PASS" if result.valid else "FAIL", (time.perf_counter() - started_at) * 1000.0)
        return result

    def _heuristic_validate(self, question_text: str, entities: list[QuestionEntity]) -> EntityValidationResult:
        question_lower = question_text.lower()
        missing_entities: list[str] = []
        mislinked: list[str] = []

        for entity in entities:
            if entity.surface.lower() not in question_lower:
                mislinked.append(f"{entity.surface}: surface not grounded in question text")

            if entity.cui is None:
                missing_entities.append(entity.surface)
            elif not self._is_valid_cui(entity.cui):
                mislinked.append(f"{entity.surface}: invalid CUI {entity.cui}")

            if entity.primekg_node_id is None:
                missing_entities.append(entity.surface)
            elif ":" not in entity.primekg_node_id or entity.primekg_node_id.endswith(":"):
                mislinked.append(f"{entity.surface}: invalid PrimeKG node {entity.primekg_node_id}")

        unique_missing = sorted(set(missing_entities))
        unique_mislinked = sorted(set(mislinked))
        issue_count = len(unique_missing) + len(unique_mislinked)
        denominator = max(len(entities) * 2, 1)
        confidence = max(0.0, min(1.0, 1.0 - (issue_count / denominator)))

        return EntityValidationResult(
            valid=issue_count == 0 and bool(entities),
            missing_entities=unique_missing,
            mislinked=unique_mislinked,
            confidence=confidence,
        )

    @staticmethod
    def _is_valid_cui(cui: str) -> bool:
        if not re.fullmatch(r"C\d{7}", cui):
            return False
        suffix = cui[1:]
        if suffix in {"0000000", "9999999"}:
            return False
        return True


class EvidenceGroundingInspector(JudgeBase):
    def __init__(self, model_name: str = DEFAULT_JUDGE_MODEL_NAME, device: str = "cuda:1") -> None:
        super().__init__(model_name=model_name, system_prompt=GROUNDING_INSPECTOR_SYSTEM_PROMPT, device=device)
        self.nli_pipeline = self._load_local_nli_pipeline()

    def evaluate(
        self,
        *,
        evidence_bundle: EvidenceBundle,
        candidate_claims: list[str] | None = None,
        candidate_answer: str | None = None,
    ) -> JudgeOutput:
        self._log_invoked("grounding")
        started_at = time.perf_counter()
        claims = self._extract_claims(candidate_claims=candidate_claims, candidate_answer=candidate_answer)
        fallback = self._heuristic_grounding(evidence_bundle=evidence_bundle, claims=claims)
        prompt = (
            f"Question: {evidence_bundle.question_text}\n"
            f"Claims: {json.dumps(claims, ensure_ascii=True)}\n"
            f"Evidence: {json.dumps(self._serialize_evidence(evidence_bundle), ensure_ascii=True)}\n"
            "Verify each claim against the evidence."
        )
        result = self._call_llm(prompt=prompt, schema=JudgeOutput, fallback_payload=fallback.model_dump())
        self._log_verdict("grounding", result.overall_recommendation, (time.perf_counter() - started_at) * 1000.0)
        for claim in result.claims:
            if claim.severity == 3 or result.h5_present:
                self.event_logger.log_event(
                    "hallucination_flag",
                    {
                        "category": "H5",
                        "severity": claim.severity,
                        "claim_text": claim.text,
                        "backend_used": self.backend,
                        "latency_ms": 0.0,
                    },
                )
        return result

    def _heuristic_grounding(self, evidence_bundle: EvidenceBundle, claims: list[str]) -> JudgeOutput:
        evidence_records = self._build_evidence_records(evidence_bundle)
        claim_verdicts: list[ClaimVerdict] = []
        for claim_text in claims:
            verdict = self._score_claim(claim_text=claim_text, evidence_records=evidence_records)
            claim_verdicts.append(verdict)

        if not claim_verdicts:
            claim_verdicts.append(
                ClaimVerdict(
                    text="No candidate claims were provided.",
                    claim_type="contextual",
                    verdict="out_of_scope",
                    evidence_id="none",
                    severity=0,
                    rationale="There was no answer content to verify.",
                )
            )

        total_supported = sum(1 for verdict in claim_verdicts if verdict.verdict == "supported")
        contradiction_penalty = sum(1 for verdict in claim_verdicts if verdict.verdict == "contradicted")
        faithfulness_score = max(0.0, min(1.0, (total_supported / len(claim_verdicts)) - (0.25 * contradiction_penalty)))
        h5_present = any(
            verdict.severity == 3 or (verdict.claim_type == "life_critical" and verdict.verdict != "supported")
            for verdict in claim_verdicts
        )
        overall_recommendation: Recommendation = "ACCEPT"
        if h5_present:
            overall_recommendation = "ABSTAIN"
        elif any(verdict.verdict in {"unsupported", "contradicted"} for verdict in claim_verdicts):
            overall_recommendation = "REVISE"

        return JudgeOutput(
            claims=claim_verdicts,
            faithfulness_score=faithfulness_score,
            h5_present=h5_present,
            overall_recommendation=overall_recommendation,
            abstention_reason=DEFAULT_ABSTENTION_REASON_H5 if h5_present else None,
        )

    def _score_claim(self, claim_text: str, evidence_records: list[dict[str, Any]]) -> ClaimVerdict:
        claim_type = self._classify_claim_type(claim_text)
        cited_evidence_ids = self._extract_evidence_refs(claim_text)
        candidate_records = [
            record for record in evidence_records if not cited_evidence_ids or record["evidence_id"] in cited_evidence_ids
        ]
        if cited_evidence_ids and not candidate_records:
            severity = self._severity_for(claim_type=claim_type, verdict="unsupported")
            return ClaimVerdict(
                text=claim_text,
                claim_type=claim_type,
                verdict="unsupported",
                evidence_id="none",
                severity=severity,
                rationale="The cited evidence identifiers are not present in the evidence bundle.",
            )

        best_record: dict[str, Any] | None = None
        best_support_score = 0.0
        best_contradiction_score = 0.0
        claim_polarity = self._polarity_for_text(claim_text)
        for record in candidate_records:
            support_score = self._support_score(claim_text=claim_text, evidence_record=record)
            contradiction_score = self._contradiction_score(
                claim_text=claim_text,
                claim_polarity=claim_polarity,
                evidence_record=record,
            )
            record_score = max(support_score, contradiction_score)
            if record_score > max(best_support_score, best_contradiction_score):
                best_record = record
                best_support_score = support_score
                best_contradiction_score = contradiction_score

        if best_record is None:
            severity = self._severity_for(claim_type=claim_type, verdict="unsupported")
            return ClaimVerdict(
                text=claim_text,
                claim_type=claim_type,
                verdict="unsupported",
                evidence_id="none",
                severity=severity,
                rationale="No relevant evidence matched the claim.",
            )

        if best_contradiction_score >= DEFAULT_CONTRADICTION_THRESHOLD and best_contradiction_score >= best_support_score:
            severity = self._severity_for(claim_type=claim_type, verdict="contradicted")
            return ClaimVerdict(
                text=claim_text,
                claim_type=claim_type,
                verdict="contradicted",
                evidence_id=best_record["evidence_id"],
                severity=severity,
                rationale=f"Evidence {best_record['evidence_id']} conflicts with the claim content.",
            )

        if best_support_score >= DEFAULT_SUPPORT_THRESHOLD:
            return ClaimVerdict(
                text=claim_text,
                claim_type=claim_type,
                verdict="supported",
                evidence_id=best_record["evidence_id"],
                severity=0,
                rationale=f"Evidence {best_record['evidence_id']} overlaps with the claim and supports it.",
            )

        severity = self._severity_for(claim_type=claim_type, verdict="unsupported")
        return ClaimVerdict(
            text=claim_text,
            claim_type=claim_type,
            verdict="unsupported",
            evidence_id="none",
            severity=severity,
            rationale="The claim could not be grounded to any KG edge or PubMed passage.",
        )

    def _support_score(self, claim_text: str, evidence_record: dict[str, Any]) -> float:
        claim_tokens = _tokenize_text(claim_text)
        evidence_tokens = _tokenize_text(str(evidence_record["text"]))
        if not claim_tokens or not evidence_tokens:
            return 0.0
        overlap = len(claim_tokens & evidence_tokens) / len(claim_tokens)
        score = overlap
        entity_hits = sum(1 for entity_text in evidence_record["entities"] if entity_text and entity_text in claim_text.lower())
        if evidence_record["entities"]:
            score += 0.15 * (entity_hits / len(evidence_record["entities"]))
        if evidence_record["evidence_id"] in self._extract_evidence_refs(claim_text):
            score += 0.1

        nli_support = self._nli_support_score(claim_text=claim_text, evidence_text=str(evidence_record["text"]))
        if nli_support > 0.0:
            score = max(score, nli_support)

        return max(0.0, min(1.0, score))

    def _contradiction_score(self, claim_text: str, claim_polarity: str, evidence_record: dict[str, Any]) -> float:
        evidence_polarity = str(evidence_record["polarity"])
        score = 0.0
        if claim_polarity != "neutral" and evidence_polarity != "neutral" and claim_polarity != evidence_polarity:
            score += 0.6
        contradiction_nli = self._nli_contradiction_score(claim_text=claim_text, evidence_text=str(evidence_record["text"]))
        score = max(score, contradiction_nli)
        lexical_overlap = len(_tokenize_text(claim_text) & _tokenize_text(str(evidence_record["text"])))
        if score > 0.0 and lexical_overlap > 0:
            score = min(1.0, score + 0.1)
        return max(0.0, min(1.0, score))

    def _build_evidence_records(self, evidence_bundle: EvidenceBundle) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for edge in evidence_bundle.subgraph_edges:
            head_text = _humanize_identifier(edge.head)
            tail_text = _humanize_identifier(edge.tail)
            text = f"{head_text} {edge.display_relation} {tail_text}"
            records.append(
                {
                    "evidence_id": f"edge:{edge.edge_id}",
                    "text": text,
                    "entities": [head_text.lower(), tail_text.lower()],
                    "polarity": self._relation_polarity(edge.relation, edge.display_relation),
                }
            )
        for passage in evidence_bundle.pubmed_passages:
            passage_text = f"{passage.title} {passage.abstract}".strip()
            records.append(
                {
                    "evidence_id": f"PMID:{passage.pmid}",
                    "text": passage_text,
                    "entities": [_entity.surface.lower() for _entity in evidence_bundle.question_entities if _entity.surface],
                    "polarity": self._polarity_for_text(passage_text),
                }
            )
        return records

    def _serialize_evidence(self, evidence_bundle: EvidenceBundle) -> dict[str, Any]:
        return {
            "subgraph_edges": [edge.model_dump() for edge in evidence_bundle.subgraph_edges],
            "pubmed_passages": [passage.model_dump() for passage in evidence_bundle.pubmed_passages],
        }

    @staticmethod
    def _extract_claims(candidate_claims: list[str] | None, candidate_answer: str | None) -> list[str]:
        if candidate_claims:
            cleaned = [claim.strip() for claim in candidate_claims if claim and claim.strip()]
            return list(dict.fromkeys(cleaned))
        if candidate_answer is None:
            return []
        fragments = re.split(r"(?<=[.!?])\s+|\n+", candidate_answer.strip())
        cleaned = [fragment.strip() for fragment in fragments if fragment and fragment.strip()]
        return list(dict.fromkeys(cleaned))

    @staticmethod
    def _classify_claim_type(claim_text: str) -> str:
        lowered = claim_text.lower()
        if "edge:" in lowered or "pmid:" in lowered or "edge_id:" in lowered:
            return "source_attribution"
        if any(term in lowered for term in DEFAULT_LIFE_CRITICAL_TERMS):
            return "life_critical"
        if any(term in lowered for term in DEFAULT_TEMPORAL_TERMS):
            return "temporal"
        if any(term in lowered for term in DEFAULT_POPULATION_TERMS):
            return "population"
        if any(term in lowered for term in DEFAULT_LOGICAL_TERMS):
            return "logical"
        if "in " in lowered or "for " in lowered:
            return "contextual"
        return "factual"

    @staticmethod
    def _severity_for(claim_type: str, verdict: str) -> int:
        if verdict == "supported" or verdict == "out_of_scope":
            return 0
        if claim_type == "life_critical":
            return 3
        if verdict == "contradicted":
            return 2
        if claim_type in {"logical", "temporal", "population"}:
            return 2
        if claim_type == "source_attribution":
            return 1
        return 1

    @staticmethod
    def _extract_evidence_refs(text: str) -> set[str]:
        normalized_refs: set[str] = set()
        for raw_ref in re.findall(r"(?:PMID:\S+|edge:\S+|edge_id:\S+)", text):
            if raw_ref.startswith("edge_id:"):
                normalized_refs.add(f"edge:{raw_ref[len('edge_id:'):]}")
            else:
                normalized_refs.add(raw_ref.rstrip(".,;:"))
        return normalized_refs

    @staticmethod
    def _relation_polarity(relation: str, display_relation: str) -> str:
        lowered_display = display_relation.lower()
        if relation in DEFAULT_RELATION_NEGATIVE_TERMS or any(term in lowered_display for term in ("contra", "risk", "adverse", "avoid")):
            return "negative"
        if any(term in lowered_display for term in DEFAULT_POSITIVE_TERMS):
            return "positive"
        return "neutral"

    @staticmethod
    def _polarity_for_text(text: str) -> str:
        lowered = text.lower()
        has_negative = any(term in lowered for term in DEFAULT_NEGATION_TERMS)
        has_positive = any(term in lowered for term in DEFAULT_POSITIVE_TERMS)
        if has_negative and not has_positive:
            return "negative"
        if has_positive and not has_negative:
            return "positive"
        return "neutral"

    def _load_local_nli_pipeline(self) -> Any | None:
        try:  # pragma: no cover - optional dependency path
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

            if torch is None:
                return None
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_NLI_MODEL_NAME, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                DEFAULT_NLI_MODEL_NAME,
                local_files_only=True,
            )
            return pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device != "cpu" and torch.cuda.is_available() else -1,
            )
        except Exception:
            return None

    def _nli_support_score(self, claim_text: str, evidence_text: str) -> float:
        if self.nli_pipeline is None:
            return 0.0
        try:  # pragma: no cover - optional dependency path
            output = self.nli_pipeline({"text": evidence_text, "text_pair": claim_text}, truncation=True)
        except Exception:
            return 0.0
        if isinstance(output, list):
            output = output[0]
        label = str(output.get("label", "")).lower()
        score = float(output.get("score", 0.0))
        if "entail" in label:
            return score
        return 0.0

    def _nli_contradiction_score(self, claim_text: str, evidence_text: str) -> float:
        if self.nli_pipeline is None:
            return 0.0
        try:  # pragma: no cover - optional dependency path
            output = self.nli_pipeline({"text": evidence_text, "text_pair": claim_text}, truncation=True)
        except Exception:
            return 0.0
        if isinstance(output, list):
            output = output[0]
        label = str(output.get("label", "")).lower()
        score = float(output.get("score", 0.0))
        if "contrad" in label:
            return score
        return 0.0


class ReasoningSoundnessAuditor(JudgeBase):
    def __init__(self, model_name: str = DEFAULT_JUDGE_MODEL_NAME, device: str = "cuda:1") -> None:
        super().__init__(model_name=model_name, system_prompt=SOUNDNESS_AUDITOR_SYSTEM_PROMPT, device=device)

    def evaluate(
        self,
        *,
        trm_output: TRMOutput,
        question: str,
        evidence_bundle: EvidenceBundle | None = None,
    ) -> ReasoningSoundnessResult:
        self._log_invoked("soundness")
        started_at = time.perf_counter()
        fallback = self._heuristic_soundness(trm_output=trm_output, question=question, evidence_bundle=evidence_bundle)
        prompt = (
            f"Question: {question}\n"
            f"TRM paths: {json.dumps([path.model_dump() for path in trm_output.ranked_paths], ensure_ascii=True)}\n"
            "Audit the reasoning chain for missing steps and unsafe logic."
        )
        result = self._call_llm(prompt=prompt, schema=ReasoningSoundnessResult, fallback_payload=fallback.model_dump())
        self._log_verdict("soundness", "PASS" if result.sound else "FAIL", (time.perf_counter() - started_at) * 1000.0)
        return result

    def _heuristic_soundness(
        self,
        trm_output: TRMOutput,
        question: str,
        evidence_bundle: EvidenceBundle | None,
    ) -> ReasoningSoundnessResult:
        gaps: list[str] = []
        expected_edges = self._expected_edge_count(question)
        top_path = trm_output.ranked_paths[0] if trm_output.ranked_paths else None
        top_path_edges = len(top_path.edges) if top_path is not None else 0

        if top_path is None:
            gaps.append("No reasoning path was produced by the TRM.")
        elif top_path_edges < expected_edges:
            gaps.append("Top-ranked reasoning path skips at least one required reasoning step.")

        if trm_output.contradiction_flag:
            gaps.append("TRM contradiction flag is active.")

        contradiction_density = 0.0
        if evidence_bundle is not None and top_path is not None:
            contradiction_density = self._path_contradiction_density(top_path_edges=top_path.edges, evidence_bundle=evidence_bundle)
            if contradiction_density > 0.5:
                gaps.append("Reasoning path traverses conflicting or medically unsafe relations.")

        rcs = 0.0
        if top_path is not None:
            coverage_ratio = min(1.0, top_path_edges / max(expected_edges, 1))
            rcs = max(0.0, min(1.0, 0.5 * coverage_ratio + 0.5 * top_path.confidence))
        rns = self._relation_node_soundness(top_path=top_path, evidence_bundle=evidence_bundle)
        cdr = max(0.0, min(1.0, 1.0 - max(contradiction_density, 1.0 if trm_output.contradiction_flag else 0.0)))
        sound = not gaps and rcs >= 0.5 and rns >= 0.5 and cdr >= 0.5

        return ReasoningSoundnessResult(sound=sound, gaps=gaps, rcs=rcs, rns=rns, cdr=cdr)

    @staticmethod
    def _expected_edge_count(question: str) -> int:
        lowered = question.lower()
        if any(term in lowered for term in ("how", "why", "interaction", "interact", "mechanism", "cause", "contraind")):
            return DEFAULT_EXPECTED_EDGE_COUNT_COMPLEX
        return DEFAULT_EXPECTED_EDGE_COUNT_SIMPLE

    @staticmethod
    def _path_contradiction_density(top_path_edges: list[str], evidence_bundle: EvidenceBundle) -> float:
        edge_lookup = {edge.edge_id: edge for edge in evidence_bundle.subgraph_edges}
        if not top_path_edges:
            return 0.0
        negative_edges = 0
        for edge_id in top_path_edges:
            edge = edge_lookup.get(edge_id)
            if edge is None:
                continue
            display_lower = edge.display_relation.lower()
            if edge.relation in DEFAULT_RELATION_NEGATIVE_TERMS or any(
                term in display_lower for term in ("contra", "risk", "adverse", "avoid")
            ):
                negative_edges += 1
        return negative_edges / len(top_path_edges)

    @staticmethod
    def _relation_node_soundness(
        top_path: Any | None,
        evidence_bundle: EvidenceBundle | None,
    ) -> float:
        if top_path is None:
            return 0.0
        if evidence_bundle is None or not evidence_bundle.question_entities:
            return 1.0 if top_path.nodes else 0.0
        question_nodes = {
            entity.primekg_node_id for entity in evidence_bundle.question_entities if entity.primekg_node_id is not None
        }
        if not question_nodes:
            return 1.0 if top_path.nodes else 0.0
        overlap = len(question_nodes & set(top_path.nodes))
        return overlap / len(question_nodes)


class AnswerFaithfulnessGuardian(JudgeBase):
    def __init__(self, model_name: str = DEFAULT_JUDGE_MODEL_NAME, device: str = "cuda:1") -> None:
        super().__init__(model_name=model_name, system_prompt=FAITHFULNESS_GUARDIAN_SYSTEM_PROMPT, device=device)

    def evaluate(
        self,
        *,
        final_answer: str,
        entity_result: EntityValidationResult | dict[str, Any],
        grounding_output: JudgeOutput | dict[str, Any],
        soundness_output: ReasoningSoundnessResult | dict[str, Any],
    ) -> FaithfulnessGuardianResult:
        self._log_invoked("faithfulness")
        started_at = time.perf_counter()
        if isinstance(entity_result, dict):
            entity_result = EntityValidationResult.model_validate(entity_result)
        if isinstance(grounding_output, dict):
            grounding_output = JudgeOutput.model_validate(grounding_output)
        if isinstance(soundness_output, dict):
            soundness_output = ReasoningSoundnessResult.model_validate(soundness_output)
        fallback = self._heuristic_guardian(
            final_answer=final_answer,
            entity_result=entity_result,
            grounding_output=grounding_output,
            soundness_output=soundness_output,
        )
        prompt = (
            f"Answer: {final_answer}\n"
            f"Entity validation: {entity_result.model_dump_json()}\n"
            f"Grounding: {grounding_output.model_dump_json()}\n"
            f"Soundness: {soundness_output.model_dump_json()}\n"
            "Return the final recommendation."
        )
        result = self._call_llm(prompt=prompt, schema=FaithfulnessGuardianResult, fallback_payload=fallback.model_dump())
        self._log_verdict("faithfulness", result.verdict, (time.perf_counter() - started_at) * 1000.0)
        return result

    def _heuristic_guardian(
        self,
        final_answer: str,
        entity_result: EntityValidationResult,
        grounding_output: JudgeOutput,
        soundness_output: ReasoningSoundnessResult,
    ) -> FaithfulnessGuardianResult:
        del final_answer
        entity_score = entity_result.confidence if entity_result.valid else entity_result.confidence * 0.5
        unsupported_count = sum(
            1 for claim in grounding_output.claims if claim.verdict in {"unsupported", "contradicted"}
        )
        grounding_score = max(0.0, grounding_output.faithfulness_score - (0.05 * unsupported_count))
        soundness_score = (soundness_output.rcs + soundness_output.rns + soundness_output.cdr) / 3.0
        if not soundness_output.sound:
            soundness_score *= 0.75

        confidence = max(
            0.0,
            min(
                1.0,
                (DEFAULT_ENTITY_WEIGHT * entity_score)
                + (DEFAULT_GROUNDING_WEIGHT * grounding_score)
                + (DEFAULT_SOUNDNESS_WEIGHT * soundness_score),
            ),
        )

        reasons: list[str] = []
        if not entity_result.valid:
            reasons.append("Entity validation reported missing or mislinked entities.")
        if grounding_output.h5_present:
            reasons.append("Blocking H5 hallucination was detected in claim grounding.")
        elif unsupported_count > 0:
            reasons.append("Some answer claims are unsupported or contradicted by the evidence.")
        if not soundness_output.sound:
            reasons.append("Reasoning soundness audit found skipped or unsafe reasoning steps.")

        verdict: Recommendation = "ACCEPT"
        if grounding_output.h5_present or confidence < DEFAULT_ABSTAIN_THRESHOLD:
            verdict = "ABSTAIN"
        elif reasons or confidence < DEFAULT_ACCEPT_THRESHOLD:
            verdict = "REVISE"

        reasoning = " ".join(reasons) if reasons else "All judge signals are aligned."
        return FaithfulnessGuardianResult(
            verdict=verdict,
            reasoning=reasoning,
            confidence=confidence,
            h5_present=grounding_output.h5_present,
        )


class HallucinationJudge:
    def __init__(
        self,
        model_name: str = DEFAULT_JUDGE_MODEL_NAME,
        device: str = "cuda:1",
        max_retries: int = DEFAULT_REVISION_RETRIES,
        abstain_threshold: float = DEFAULT_ABSTAIN_THRESHOLD,
    ) -> None:
        self.max_retries = max_retries
        self.abstain_threshold = abstain_threshold
        self.event_logger = StructuredLogger("layer4_judges", DEFAULT_LOG_DIR)
        self.entity_validator = EntityValidator(model_name=model_name, device=device)
        self.grounding_inspector = EvidenceGroundingInspector(model_name=model_name, device=device)
        self.soundness_auditor = ReasoningSoundnessAuditor(model_name=model_name, device=device)
        self.faithfulness_guardian = AnswerFaithfulnessGuardian(model_name=model_name, device=device)

    def evaluate(
        self,
        *,
        final_answer: str,
        evidence_bundle: EvidenceBundle,
        trm_output: TRMOutput,
        candidate_claims: list[str] | None = None,
    ) -> HallucinationJudgeResult:
        started_at = time.perf_counter()
        revisions: list[str] = []
        retries_used = 0
        current_answer = final_answer.strip()

        entity_result = self.entity_validator.evaluate(
            question_text=evidence_bundle.question_text,
            entities=evidence_bundle.question_entities,
        )
        soundness_output = self.soundness_auditor.evaluate(
            trm_output=trm_output,
            question=evidence_bundle.question_text,
            evidence_bundle=evidence_bundle,
        )

        grounding_output = self.grounding_inspector.evaluate(
            evidence_bundle=evidence_bundle,
            candidate_claims=candidate_claims,
            candidate_answer=current_answer if candidate_claims is None else None,
        )
        faithfulness_output = self.faithfulness_guardian.evaluate(
            final_answer=current_answer,
            entity_result=entity_result,
            grounding_output=grounding_output,
            soundness_output=soundness_output,
        )

        while faithfulness_output.verdict == "REVISE" and retries_used < self.max_retries:
            revised_answer = self._revise_answer(current_answer=current_answer, grounding_output=grounding_output)
            if not revised_answer or revised_answer == current_answer:
                break
            revisions.append(revised_answer)
            retries_used += 1
            current_answer = revised_answer
            grounding_output = self.grounding_inspector.evaluate(
                evidence_bundle=evidence_bundle,
                candidate_claims=None,
                candidate_answer=current_answer,
            )
            faithfulness_output = self.faithfulness_guardian.evaluate(
                final_answer=current_answer,
                entity_result=entity_result,
                grounding_output=grounding_output,
                soundness_output=soundness_output,
            )

        final_recommendation = faithfulness_output.verdict
        abstention_reason: str | None = None
        if grounding_output.h5_present or faithfulness_output.confidence < self.abstain_threshold:
            final_recommendation = "ABSTAIN"
            abstention_reason = grounding_output.abstention_reason or DEFAULT_ABSTENTION_REASON_H5
        elif faithfulness_output.verdict == "ABSTAIN":
            abstention_reason = grounding_output.abstention_reason or faithfulness_output.reasoning
        elif grounding_output.overall_recommendation == "ABSTAIN":
            abstention_reason = grounding_output.abstention_reason

        parse_success_rate = self._aggregate_parse_success_rate()
        result = HallucinationJudgeResult(
            answer_text=current_answer,
            final_recommendation=final_recommendation,
            confidence=faithfulness_output.confidence,
            retries_used=retries_used,
            h5_present=grounding_output.h5_present or faithfulness_output.h5_present,
            abstention_reason=abstention_reason,
            entity_validation=entity_result,
            grounding_output=grounding_output,
            soundness_output=soundness_output,
            faithfulness_output=faithfulness_output,
            revisions=revisions,
            parse_success_rate=parse_success_rate,
        )
        self.event_logger.log_event(
            "judge_verdict",
            {
                "judge_type": "hallucination_wrapper",
                "verdict": result.final_recommendation,
                "backend_used": {
                    "entity": self.entity_validator.backend,
                    "grounding": self.grounding_inspector.backend,
                    "soundness": self.soundness_auditor.backend,
                    "faithfulness": self.faithfulness_guardian.backend,
                },
                "latency_ms": (time.perf_counter() - started_at) * 1000.0,
            },
        )
        if result.h5_present:
            flagged_claim = next((claim for claim in grounding_output.claims if claim.severity == 3), None)
            self.event_logger.log_event(
                "hallucination_flag",
                {
                    "category": "H5",
                    "severity": flagged_claim.severity if flagged_claim is not None else 3,
                    "claim_text": flagged_claim.text if flagged_claim is not None else "unknown",
                    "backend_used": "wrapper",
                    "latency_ms": 0.0,
                },
            )
        return result

    def _aggregate_parse_success_rate(self) -> float:
        judges = (
            self.entity_validator,
            self.grounding_inspector,
            self.soundness_auditor,
            self.faithfulness_guardian,
        )
        total_attempts = sum(judge.parse_attempts for judge in judges)
        total_successes = sum(judge.parse_successes for judge in judges)
        if total_attempts == 0:
            return 1.0
        return total_successes / total_attempts

    @staticmethod
    def _revise_answer(current_answer: str, grounding_output: JudgeOutput) -> str:
        supported_claims = [claim for claim in grounding_output.claims if claim.verdict == "supported"]
        if not supported_claims:
            return current_answer
        revised_sentences: list[str] = []
        for claim in supported_claims:
            sentence = claim.text
            if claim.evidence_id != "none" and claim.evidence_id not in sentence:
                sentence = f"{sentence} [{claim.evidence_id}]"
            revised_sentences.append(sentence)
        return " ".join(revised_sentences)


def _humanize_identifier(identifier: str) -> str:
    normalized = identifier.split(":", 1)[-1]
    return normalized.replace("_", " ").replace("-", " ").strip() or identifier


def _tokenize_text(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if token not in DEFAULT_STOPWORDS and len(token) > 1}


__all__ = [
    "AnswerFaithfulnessGuardian",
    "EntityValidationResult",
    "EntityValidator",
    "EvidenceGroundingInspector",
    "FaithfulnessGuardianResult",
    "HallucinationJudge",
    "HallucinationJudgeResult",
    "JudgeBase",
    "ReasoningSoundnessAuditor",
    "ReasoningSoundnessResult",
]


def _resolve_local_model_path(model_name: str) -> str:
    candidate = Path(model_name)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    if str(model_name).replace("\\", "/").startswith("data/"):
        resolved = KaggleEnv.path(model_name)
        if resolved.exists():
            return str(resolved)
    return model_name


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
