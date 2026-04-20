from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.layers.layer4_judges import DEFAULT_MAX_LLM_RETRIES, JudgeBase
from src.schemas import EvidenceBundle
from schemas.posthoc_auditor_schema import (
    PostHocAuditClaim,
    PostHocAuditMode,
    PostHocAuditOutput,
    PostHocReasoningGap,
)


DEFAULT_POSTHOC_AUDITOR_MODEL = "heuristic"
DEFAULT_SUPPORT_THRESHOLD = 0.30
DEFAULT_CONTRADICTION_THRESHOLD = 0.35
DEFAULT_LIFE_CRITICAL_TERMS = {
    "dose",
    "dosage",
    "mg",
    "mcg",
    "contraindication",
    "contraindicated",
    "interaction",
    "ddi",
    "bleeding",
    "hemorrhage",
    "safe",
    "unsafe",
    "avoid",
}
DEFAULT_NEGATION_TERMS = {
    "no",
    "not",
    "never",
    "without",
    "avoid",
    "unsafe",
    "contraindicated",
    "risk",
    "adverse",
}
DEFAULT_POSITIVE_TERMS = {
    "safe",
    "recommended",
    "effective",
    "beneficial",
    "indicated",
    "reduces",
    "prevents",
    "treats",
}
DEFAULT_REASONING_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
DEFAULT_EDGE_REF_PATTERN = re.compile(r"(?:edge:|edge_id:)\s*([A-Za-z0-9_.:-]+)|edgeid\s*([A-Za-z0-9_.:-]+)", re.IGNORECASE)
DEFAULT_PMID_REF_PATTERN = re.compile(r"PMID:\s*([0-9]+)", re.IGNORECASE)
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
    "that",
    "the",
    "to",
    "with",
}
PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "posthoc_auditor_system.txt"


@dataclass(slots=True)
class PostHocAuditorResultBundle:
    output: PostHocAuditOutput
    latency_ms: float
    parse_success: bool
    json_retry_count: int
    auditor_model: str

    def to_json_dict(self) -> dict[str, Any]:
        payload = self.output.model_dump(mode="json")
        payload.setdefault("audit_metadata", {})
        payload["audit_metadata"].update(
            {
                "auditor_model": self.auditor_model,
                "latency_ms": self.latency_ms,
                "json_retry_count": self.json_retry_count,
                "parse_success": self.parse_success,
            }
        )
        return payload


class PostHocAuditor(JudgeBase):
    def __init__(self, model_name: str = DEFAULT_POSTHOC_AUDITOR_MODEL, device: str = "cpu") -> None:
        system_prompt = _load_system_prompt()
        super().__init__(model_name=model_name, system_prompt=system_prompt, device=device)

    def evaluate(self, **kwargs: Any) -> PostHocAuditOutput:
        return self.evaluate_with_diagnostics(**kwargs).output

    def evaluate_with_diagnostics(
        self,
        *,
        question: str,
        evidence_bundle: EvidenceBundle,
        candidate_answer: str | None = None,
        current_reasoning_output: str | None = None,
        gold_cot_steps: Sequence[str] | None = None,
        hard_constraints: Mapping[str, Any] | None = None,
        mode: PostHocAuditMode = "answer_only",
    ) -> PostHocAuditorResultBundle:
        started_at = time.perf_counter()
        answer_text = _resolve_answer_text(
            candidate_answer=candidate_answer,
            current_reasoning_output=current_reasoning_output,
            gold_cot_steps=gold_cot_steps,
        )
        claim_texts = _extract_claim_texts(
            answer_text=answer_text,
            gold_cot_steps=gold_cot_steps,
            mode=mode,
        )

        if self.backend == "heuristic":
            deterministic = self._deterministic_audit(
                question=question,
                evidence_bundle=evidence_bundle,
                claim_texts=claim_texts,
                hard_constraints=hard_constraints,
                force_revise=False,
            )
            deterministic.audit_metadata.update(
                {
                    "auditor_model": self.model_name,
                    "latency_ms": (time.perf_counter() - started_at) * 1000.0,
                    "json_retry_count": 0,
                    "parse_success": True,
                    "mode": mode,
                    "backend_used": self.backend,
                }
            )
            self._log_audit_event(deterministic)
            return PostHocAuditorResultBundle(
                output=deterministic,
                latency_ms=float(deterministic.audit_metadata["latency_ms"]),
                parse_success=True,
                json_retry_count=0,
                auditor_model=self.model_name,
            )

        prompt = self._build_prompt(
            question=question,
            answer_text=answer_text,
            claim_texts=claim_texts,
            evidence_bundle=evidence_bundle,
            hard_constraints=hard_constraints,
            mode=mode,
            gold_cot_steps=gold_cot_steps,
        )
        parsed_output: PostHocAuditOutput | None = None
        last_error: Exception | None = None
        json_retry_count = 0

        for attempt in range(DEFAULT_MAX_LLM_RETRIES):
            try:
                raw_output = self._generate_structured_text(prompt)
                payload = self._extract_json_payload(raw_output)
                parsed_output = PostHocAuditOutput.model_validate(payload)
                json_retry_count = attempt
                break
            except Exception as exc:  # pragma: no cover - exercised with remote/local LLM backends
                last_error = exc
                json_retry_count = attempt + 1

        parse_success = parsed_output is not None
        if not parse_success:
            parsed_output = self._deterministic_audit(
                question=question,
                evidence_bundle=evidence_bundle,
                claim_texts=claim_texts,
                hard_constraints=hard_constraints,
                force_revise=True,
            )
            parsed_output.reasoning_gaps.append(
                PostHocReasoningGap(
                    step=f"LLM JSON parsing failed after {DEFAULT_MAX_LLM_RETRIES} attempts; fallback parser used.",
                    severity=1,
                )
            )
            if parsed_output.overall_recommendation != "ABSTAIN":
                parsed_output.overall_recommendation = "REVISE"
                parsed_output.sample_label = "partial"
            if last_error is not None:
                parsed_output.audit_metadata["last_parse_error"] = str(last_error)

        parsed_output.audit_metadata.update(
            {
                "auditor_model": self.model_name,
                "latency_ms": (time.perf_counter() - started_at) * 1000.0,
                "json_retry_count": json_retry_count,
                "parse_success": parse_success,
                "mode": mode,
                "backend_used": self.backend,
            }
        )
        self._log_audit_event(parsed_output)
        return PostHocAuditorResultBundle(
            output=parsed_output,
            latency_ms=float(parsed_output.audit_metadata["latency_ms"]),
            parse_success=parse_success,
            json_retry_count=json_retry_count,
            auditor_model=self.model_name,
        )

    def _build_prompt(
        self,
        *,
        question: str,
        answer_text: str,
        claim_texts: Sequence[str],
        evidence_bundle: EvidenceBundle,
        hard_constraints: Mapping[str, Any] | None,
        mode: PostHocAuditMode,
        gold_cot_steps: Sequence[str] | None,
    ) -> str:
        evidence_payload = {
            "primekg_edges": [edge.model_dump(mode="json") for edge in evidence_bundle.subgraph_edges],
            "pubmed_passages": [passage.model_dump(mode="json") for passage in evidence_bundle.pubmed_passages],
        }
        schema_instruction = json.dumps(PostHocAuditOutput.model_json_schema(), indent=2, sort_keys=True)
        return (
            f"Mode: {mode}\n"
            f"Question:\n{question}\n\n"
            f"Candidate answer / reasoning:\n{answer_text}\n\n"
            f"Atomic claim candidates:\n{json.dumps(list(claim_texts), ensure_ascii=True)}\n\n"
            f"Hard constraints:\n{json.dumps(dict(hard_constraints or {}), ensure_ascii=True)}\n\n"
            f"Gold CoT steps:\n{json.dumps(list(gold_cot_steps or []), ensure_ascii=True)}\n\n"
            f"Evidence bundle:\n{json.dumps(evidence_payload, ensure_ascii=True)}\n\n"
            "Return strict JSON only with this schema:\n"
            f"{schema_instruction}"
        )

    def _deterministic_audit(
        self,
        *,
        question: str,
        evidence_bundle: EvidenceBundle,
        claim_texts: Sequence[str],
        hard_constraints: Mapping[str, Any] | None,
        force_revise: bool,
    ) -> PostHocAuditOutput:
        evidence_records = _build_evidence_records(evidence_bundle)
        available_evidence_ids = {record["evidence_id"] for record in evidence_records}
        unsupported_references: set[str] = set()
        claims: list[PostHocAuditClaim] = []
        reasoning_gaps: list[PostHocReasoningGap] = []
        insufficient_evidence = not evidence_records
        lifecritical_context = _is_life_critical_context(question=question, hard_constraints=hard_constraints)

        if insufficient_evidence:
            reasoning_gaps.append(
                PostHocReasoningGap(
                    step="Evidence bundle is empty or too sparse to support reliable audit decisions.",
                    severity=2,
                )
            )

        for claim_text in claim_texts:
            normalized_claim = claim_text.strip()
            if not normalized_claim:
                continue
            claim_type = _classify_claim_type(normalized_claim)
            cited_ids = _extract_evidence_references(normalized_claim)
            unknown_refs = {evidence_id for evidence_id in cited_ids if evidence_id not in available_evidence_ids}
            if unknown_refs:
                unsupported_references.update(unknown_refs)
                severity = 3 if claim_type == "lifecritical" or lifecritical_context else 2
                claims.append(
                    PostHocAuditClaim(
                        text=normalized_claim,
                        type="sourceattribution",
                        verdict="unsupported",
                        evidence_id="none",
                        severity=severity,
                        rationale="Claim cites edge/PMID not found in the retrieved evidence bundle.",
                    )
                )
                continue

            matched_records = [
                record for record in evidence_records if not cited_ids or record["evidence_id"] in cited_ids
            ]
            claim_verdict, evidence_id, rationale = _score_claim_against_evidence(
                claim_text=normalized_claim,
                claim_type=claim_type,
                evidence_records=matched_records,
            )
            severity = _severity_for_claim(
                claim_type=claim_type,
                verdict=claim_verdict,
                lifecritical_context=lifecritical_context,
            )
            claims.append(
                PostHocAuditClaim(
                    text=normalized_claim,
                    type=claim_type,
                    verdict=claim_verdict,
                    evidence_id=evidence_id,
                    severity=severity,
                    rationale=rationale,
                )
            )

        if not claims:
            claims.append(
                PostHocAuditClaim(
                    text="No answer or reasoning content was available to audit.",
                    type="contextual",
                    verdict="out_of_scope",
                    evidence_id="none",
                    severity=1 if insufficient_evidence else 0,
                    rationale="Auditor could not extract atomic claims from provided content.",
                )
            )
            reasoning_gaps.append(
                PostHocReasoningGap(
                    step="No atomic claims could be extracted for grounding checks.",
                    severity=2 if insufficient_evidence else 1,
                )
            )

        unsupported_count = sum(1 for claim in claims if claim.verdict == "unsupported")
        contradicted_count = sum(1 for claim in claims if claim.verdict == "contradicted")
        supported_count = sum(1 for claim in claims if claim.verdict == "supported")
        total_claims = max(len(claims), 1)
        faithfulness_score = _clamp(
            (supported_count / total_claims) - (0.25 * contradicted_count) - (0.10 * unsupported_count),
            0.0,
            1.0,
        )

        if unsupported_references:
            reasoning_gaps.append(
                PostHocReasoningGap(
                    step="Answer includes unsupported source references outside the evidence bundle.",
                    severity=2,
                )
            )
        if unsupported_count > 0:
            reasoning_gaps.append(
                PostHocReasoningGap(
                    step=f"{unsupported_count} claim(s) are unsupported by PrimeKG/PubMed evidence.",
                    severity=1,
                )
            )
        if contradicted_count > 0:
            reasoning_gaps.append(
                PostHocReasoningGap(
                    step=f"{contradicted_count} claim(s) contradict retrieved evidence.",
                    severity=2,
                )
            )

        h5_present = any(claim.severity == 3 for claim in claims)
        if h5_present:
            recommendation = "ABSTAIN"
            abstention_reason = "Life-critical unsupported/contradicted claim detected."
        elif force_revise:
            recommendation = "REVISE"
            abstention_reason = None
        elif insufficient_evidence and supported_count == 0:
            recommendation = "REVISE"
            abstention_reason = None
        elif unsupported_count > 0 or contradicted_count > 0:
            recommendation = "REVISE"
            abstention_reason = None
        else:
            recommendation = "ACCEPT"
            abstention_reason = None

        return PostHocAuditOutput(
            claims=claims,
            reasoning_gaps=reasoning_gaps,
            unsupported_evidence_references=sorted(unsupported_references),
            faithfulness_score=faithfulness_score,
            h5_present=h5_present,
            overall_recommendation=recommendation,
            abstention_reason=abstention_reason,
            sample_label="partial",
            audit_metadata={},
        )

    def _log_audit_event(self, output: PostHocAuditOutput) -> None:
        metadata = output.audit_metadata
        self.event_logger.log_event(
            "posthoc_audit_complete",
            {
                "auditor_model": metadata.get("auditor_model", self.model_name),
                "latency_ms": metadata.get("latency_ms"),
                "json_retry_count": metadata.get("json_retry_count"),
                "parse_success": metadata.get("parse_success"),
                "overall_recommendation": output.overall_recommendation,
                "sample_label": output.sample_label,
                "faithfulness_score": output.faithfulness_score,
                "h5_present": output.h5_present,
                "backend_used": metadata.get("backend_used", self.backend),
            },
        )


def render_posthoc_audit_markdown(
    *,
    sample_id: str,
    question: str,
    output: PostHocAuditOutput,
) -> str:
    lines = [
        f"# Post-hoc Audit: {sample_id}",
        "",
        f"Question: {question}",
        "",
        f"Recommendation: **{output.overall_recommendation}**",
        f"Sample label: **{output.sample_label}**",
        f"Faithfulness score: `{output.faithfulness_score:.3f}`",
        f"H5 present: `{output.h5_present}`",
        "",
        "| Claim | Type | Verdict | Evidence | Severity |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for claim in output.claims:
        lines.append(
            "| {text} | {type} | {verdict} | {evidence} | {severity} |".format(
                text=claim.text.replace("|", "\\|"),
                type=claim.type,
                verdict=claim.verdict,
                evidence=claim.evidence_id,
                severity=claim.severity,
            )
        )
        lines.append(f"Rationale: {claim.rationale}")

    if output.reasoning_gaps:
        lines.extend(["", "## Reasoning Gaps", ""])
        for gap in output.reasoning_gaps:
            lines.append(f"- (sev {gap.severity}) {gap.step}")

    if output.unsupported_evidence_references:
        lines.extend(["", "## Unsupported References", ""])
        for reference in output.unsupported_evidence_references:
            lines.append(f"- {reference}")

    lines.extend(["", "## Metadata", ""])
    for key, value in sorted(output.audit_metadata.items()):
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def _load_system_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8").strip()
    return (
        "You are a post-hoc medical auditor. Use only supplied evidence. "
        "Return strict JSON only."
    )


def _resolve_answer_text(
    *,
    candidate_answer: str | None,
    current_reasoning_output: str | None,
    gold_cot_steps: Sequence[str] | None,
) -> str:
    if candidate_answer and candidate_answer.strip():
        return candidate_answer.strip()
    if current_reasoning_output and current_reasoning_output.strip():
        return current_reasoning_output.strip()
    if gold_cot_steps:
        joined = " ".join(step.strip() for step in gold_cot_steps if str(step).strip())
        if joined:
            return joined
    return ""


def _extract_claim_texts(
    *,
    answer_text: str,
    gold_cot_steps: Sequence[str] | None,
    mode: PostHocAuditMode,
) -> list[str]:
    claims: list[str] = []
    if answer_text:
        claims.extend(fragment.strip() for fragment in DEFAULT_REASONING_SPLIT_PATTERN.split(answer_text) if fragment.strip())
    if mode == "answer_plus_cot" and gold_cot_steps:
        claims.extend(str(step).strip() for step in gold_cot_steps if str(step).strip())
    deduped: list[str] = []
    for claim in claims:
        if claim and claim not in deduped:
            deduped.append(claim)
    return deduped


def _build_evidence_records(evidence_bundle: EvidenceBundle) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for edge in evidence_bundle.subgraph_edges:
        head = _humanize_identifier(edge.head)
        tail = _humanize_identifier(edge.tail)
        edge_text = f"{head} {edge.display_relation} {tail}"
        records.append(
            {
                "evidence_id": f"edge:{edge.edge_id}",
                "text": edge_text,
                "tokens": _tokenize(edge_text),
                "polarity": _polarity(edge_text),
            }
        )
    for passage in evidence_bundle.pubmed_passages:
        text = f"{passage.title} {passage.abstract}".strip()
        records.append(
            {
                "evidence_id": f"PMID:{passage.pmid}",
                "text": text,
                "tokens": _tokenize(text),
                "polarity": _polarity(text),
            }
        )
    return records


def _classify_claim_type(claim_text: str) -> str:
    lowered = claim_text.lower()
    if "pmid:" in lowered or "edge:" in lowered or "edgeid" in lowered or "edge_id:" in lowered:
        return "sourceattribution"
    if any(term in lowered for term in ("dose", "dosage", "mg", "mcg", "contraindication", "contraindicated", "interaction", "ddi")):
        return "lifecritical"
    if any(term in lowered for term in ("because", "therefore", "thus", "hence", "so ")):
        return "logical"
    if any(term in lowered for term in ("today", "tomorrow", "week", "month", "year", "acute", "chronic")):
        return "temporal"
    if any(term in lowered for term in ("adult", "child", "children", "pediatric", "pregnant", "elderly", "female", "male")):
        return "population"
    if any(term in lowered for term in ("for ", "in ", "with ", "without ")):
        return "contextual"
    return "factual"


def _extract_evidence_references(text: str) -> set[str]:
    refs: set[str] = set()
    for match in DEFAULT_EDGE_REF_PATTERN.finditer(text):
        edge_id = (match.group(1) or match.group(2) or "").strip().rstrip(".,;:")
        if edge_id:
            refs.add(f"edge:{edge_id}")
    for match in DEFAULT_PMID_REF_PATTERN.finditer(text):
        pmid = (match.group(1) or "").strip().rstrip(".,;:")
        if pmid:
            refs.add(f"PMID:{pmid}")
    return refs


def _score_claim_against_evidence(
    *,
    claim_text: str,
    claim_type: str,
    evidence_records: Sequence[Mapping[str, Any]],
) -> tuple[str, str, str]:
    if not evidence_records:
        return "unsupported", "none", "No evidence records were provided for grounding."

    claim_tokens = _tokenize(claim_text)
    if not claim_tokens:
        return "out_of_scope", "none", "Claim text is empty after tokenization."
    claim_polarity = _polarity(claim_text)

    best_support = 0.0
    best_contradiction = 0.0
    best_record_id = "none"
    for record in evidence_records:
        record_tokens = set(record.get("tokens", set()))
        if not record_tokens:
            continue
        overlap = len(claim_tokens & record_tokens) / max(len(claim_tokens), 1)
        support_score = overlap
        if claim_type == "sourceattribution" and record.get("evidence_id") in _extract_evidence_references(claim_text):
            support_score = max(support_score, 0.9)
        contradiction_score = 0.0
        record_polarity = str(record.get("polarity", "neutral"))
        if claim_polarity != "neutral" and record_polarity != "neutral" and claim_polarity != record_polarity:
            contradiction_score = max(contradiction_score, min(1.0, overlap + 0.2))
        if support_score >= best_support or contradiction_score >= best_contradiction:
            if max(support_score, contradiction_score) >= max(best_support, best_contradiction):
                best_support = support_score
                best_contradiction = contradiction_score
                best_record_id = str(record.get("evidence_id") or "none")

    if best_contradiction >= DEFAULT_CONTRADICTION_THRESHOLD and best_contradiction >= best_support:
        return (
            "contradicted",
            best_record_id,
            f"Evidence {best_record_id} conflicts with claim polarity/content.",
        )
    if best_support >= DEFAULT_SUPPORT_THRESHOLD:
        return (
            "supported",
            best_record_id,
            f"Evidence {best_record_id} overlaps with and supports the claim.",
        )
    return "unsupported", "none", "Claim is not grounded strongly enough by retrieved evidence."


def _severity_for_claim(*, claim_type: str, verdict: str, lifecritical_context: bool) -> int:
    if verdict in {"supported", "out_of_scope"}:
        return 0
    if claim_type == "lifecritical" or lifecritical_context:
        return 3
    if claim_type == "sourceattribution":
        return 2
    if verdict == "contradicted":
        return 2
    if claim_type in {"logical", "temporal", "population"}:
        return 2
    return 1


def _is_life_critical_context(*, question: str, hard_constraints: Mapping[str, Any] | None) -> bool:
    lowered_question = question.lower()
    if any(term in lowered_question for term in DEFAULT_LIFE_CRITICAL_TERMS):
        return True
    if not hard_constraints:
        return False
    joined_constraints = json.dumps(dict(hard_constraints), ensure_ascii=True).lower()
    return any(term in joined_constraints for term in DEFAULT_LIFE_CRITICAL_TERMS)


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if token not in DEFAULT_STOPWORDS}


def _polarity(text: str) -> str:
    lowered = text.lower()
    has_negative = any(term in lowered for term in DEFAULT_NEGATION_TERMS)
    has_positive = any(term in lowered for term in DEFAULT_POSITIVE_TERMS)
    if has_negative and not has_positive:
        return "negative"
    if has_positive and not has_negative:
        return "positive"
    return "neutral"


def _humanize_identifier(identifier: str) -> str:
    normalized = identifier.split(":", 1)[-1]
    return normalized.replace("_", " ").replace("-", " ").strip() or identifier


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

