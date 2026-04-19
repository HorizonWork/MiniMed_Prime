from __future__ import annotations

import logging
import math
import re
from itertools import combinations
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())


DEFAULT_BOOTSTRAP_SEED = 13
DEFAULT_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
DEFAULT_NUMERIC_REL_TOL = 0.05
DEFAULT_NUMERIC_ABS_TOL = 0.5
DEFAULT_ENTAILMENT_THRESHOLD = 0.5
DEFAULT_CONTRADICTION_THRESHOLD = 0.5
DEFAULT_LEXICAL_SUPPORT_THRESHOLD = 0.35
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
DEFAULT_POSITIVE_CUES = {
    "beneficial",
    "effective",
    "helps",
    "improves",
    "indicated",
    "prevents",
    "recommended",
    "reduces",
    "safe",
    "supports",
    "treats",
}
DEFAULT_NEGATIVE_CUES = {
    "avoid",
    "contraindicated",
    "harmful",
    "increases risk",
    "not",
    "unsafe",
    "worsens",
}
DEFAULT_PROVENANCE_PATTERN = re.compile(r"\[(edge:[^\]]+|edge_id:[^\]]+|PMID:[^\]]+)\]")
DEFAULT_CLAIM_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
DEFAULT_NUMERIC_PATTERN = re.compile(
    r"\b(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mcg|mg|g|kg|ml|mL|L|units|%|mmhg|mmol/?l|mg/?dl|g/?day|mg/?day|mcg/?day)?\b",
    flags=re.IGNORECASE,
)
DEFAULT_ABBREVIATION_PATTERN = re.compile(r"\b(?P<long>[A-Za-z][A-Za-z \-]{2,})\s*\((?P<short>[A-Za-z0-9\-]{2,10})\)")
DEFAULT_LABEL_SYNONYMS = {
    "yes": {"yes", "y", "true", "correct", "supported", "affirmative"},
    "no": {"no", "n", "false", "incorrect", "unsupported", "negative"},
    "maybe": {"maybe", "uncertain", "unknown", "inconclusive", "possibly", "possible"},
}
DEFAULT_UMLS_EXPANSIONS = {
    "acei": "angiotensin converting enzyme inhibitor",
    "af": "atrial fibrillation",
    "aki": "acute kidney injury",
    "arb": "angiotensin receptor blocker",
    "bid": "twice daily",
    "cabg": "coronary artery bypass grafting",
    "cad": "coronary artery disease",
    "ckd": "chronic kidney disease",
    "copd": "chronic obstructive pulmonary disease",
    "dm": "diabetes mellitus",
    "dvt": "deep vein thrombosis",
    "hf": "heart failure",
    "htn": "hypertension",
    "mi": "myocardial infarction",
    "nsaid": "nonsteroidal anti inflammatory drug",
    "pe": "pulmonary embolism",
    "qd": "once daily",
    "ra": "rheumatoid arthritis",
    "tid": "three times daily",
}


def exact_match(predicted: str, gold: str) -> float:
    predicted_normalized = _normalize_answer_text(predicted)
    gold_normalized = _normalize_answer_text(gold)
    return 1.0 if predicted_normalized == gold_normalized else 0.0


def macro_f1(predictions: Sequence[str], golds: Sequence[str]) -> float:
    if len(predictions) != len(golds):
        raise ValueError("predictions and golds must have the same length.")
    if not predictions:
        return 0.0

    canonical_predictions = [_canonical_label(prediction) for prediction in predictions]
    canonical_golds = [_canonical_label(gold) for gold in golds]
    label_space = sorted(set(canonical_predictions) | set(canonical_golds))
    if not label_space:
        return 0.0

    per_label_f1: list[float] = []
    for label in label_space:
        true_positive = sum(
            1 for predicted, gold in zip(canonical_predictions, canonical_golds) if predicted == label and gold == label
        )
        false_positive = sum(
            1 for predicted, gold in zip(canonical_predictions, canonical_golds) if predicted == label and gold != label
        )
        false_negative = sum(
            1 for predicted, gold in zip(canonical_predictions, canonical_golds) if predicted != label and gold == label
        )
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        per_label_f1.append(_f1_from_precision_recall(precision, recall))
    return float(sum(per_label_f1) / len(per_label_f1))


def ragas_faithfulness(answer: str, context: str | Sequence[str], nli_model: Any | None) -> float:
    claims = extract_claims(answer)
    if not claims:
        return 1.0

    context_chunks = _context_to_chunks(context)
    if not context_chunks:
        return 0.0

    abbreviation_dict = _build_abbreviation_dictionary([answer, *context_chunks])
    expanded_context = [_normalize_medical_text(chunk, abbreviation_dict) for chunk in context_chunks]
    supported_claims = 0
    for claim in claims:
        expanded_claim = _normalize_medical_text(claim, abbreviation_dict)
        if _claim_supported(expanded_claim, expanded_context, nli_model):
            supported_claims += 1
    return _safe_divide(supported_claims, len(claims))


def f1_hallucination(
    claims_supported: int | Sequence[str],
    total_claims: int | Sequence[str],
    gold_claims: int | Sequence[str],
) -> tuple[float, float, float]:
    if isinstance(claims_supported, int) and isinstance(total_claims, int) and isinstance(gold_claims, int):
        precision = _safe_divide(claims_supported, total_claims)
        recall = _safe_divide(claims_supported, gold_claims)
        return precision, recall, _f1_from_precision_recall(precision, recall)

    supported_set = {_normalize_claim_identifier(claim) for claim in _ensure_sequence(claims_supported)}
    predicted_set = {_normalize_claim_identifier(claim) for claim in _ensure_sequence(total_claims)}
    gold_set = {_normalize_claim_identifier(claim) for claim in _ensure_sequence(gold_claims)}
    matched_supported = supported_set & gold_set if gold_set else supported_set
    precision = _safe_divide(len(matched_supported), len(predicted_set))
    recall = _safe_divide(len(matched_supported), len(gold_set))
    return precision, recall, _f1_from_precision_recall(precision, recall)


def contradiction_detection_rate(claims: Sequence[Any], nli_model: Any | None) -> float:
    if not claims:
        return 0.0

    if all(isinstance(item, dict) for item in claims):
        contradiction_flags = []
        for item in claims:
            claim_text = str(item.get("claim", item.get("hypothesis", "")))
            context_text = str(item.get("context", item.get("premise", "")))
            if not claim_text or not context_text:
                continue
            contradiction_flags.append(_claim_contradicted(claim_text, context_text, nli_model))
        if not contradiction_flags:
            return 0.0
        return _safe_divide(sum(1 for flag in contradiction_flags if flag), len(contradiction_flags))

    normalized_claims = [str(claim).strip() for claim in claims if str(claim).strip()]
    pairwise_flags = [
        _pairwise_contradiction(first_claim, second_claim, nli_model)
        for first_claim, second_claim in combinations(normalized_claims, 2)
    ]
    if not pairwise_flags:
        return 0.0
    return _safe_divide(sum(1 for flag in pairwise_flags if flag), len(pairwise_flags))


def reasoning_completeness_score(predicted_edges: Sequence[str], gold_edges: Sequence[str]) -> float:
    predicted_set = {_normalize_edge_identifier(edge) for edge in predicted_edges if str(edge).strip()}
    gold_set = {_normalize_edge_identifier(edge) for edge in gold_edges if str(edge).strip()}
    if not gold_set:
        return 1.0
    return _safe_divide(len(predicted_set & gold_set), len(gold_set))


def reasoning_necessity_score(predicted_edges: Sequence[str], gold_edges: Sequence[str]) -> float:
    predicted_set = {_normalize_edge_identifier(edge) for edge in predicted_edges if str(edge).strip()}
    gold_set = {_normalize_edge_identifier(edge) for edge in gold_edges if str(edge).strip()}
    if not predicted_set:
        return 1.0 if not gold_set else 0.0
    return _safe_divide(len(predicted_set & gold_set), len(predicted_set))


def mcnemar_test(model_a_correct: Sequence[bool], model_b_correct: Sequence[bool]) -> tuple[float, float]:
    if len(model_a_correct) != len(model_b_correct):
        raise ValueError("model_a_correct and model_b_correct must have the same length.")
    if not model_a_correct:
        return 0.0, 1.0

    b_count = sum(1 for a_correct, b_correct in zip(model_a_correct, model_b_correct) if a_correct and not b_correct)
    c_count = sum(1 for a_correct, b_correct in zip(model_a_correct, model_b_correct) if (not a_correct) and b_correct)
    discordant = b_count + c_count
    if discordant == 0:
        return 0.0, 1.0

    chi_square = ((abs(b_count - c_count) - 1) ** 2) / discordant
    p_value = math.erfc(math.sqrt(max(chi_square, 0.0) / 2.0))
    return float(chi_square), float(p_value)


def bootstrap_ci(
    differences: Sequence[float],
    n_bootstrap: int = 1000,
    confidence_level: float = DEFAULT_BOOTSTRAP_CONFIDENCE_LEVEL,
    random_state: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float]:
    values = np.asarray(list(differences), dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    if values.size == 1:
        scalar = float(values[0])
        return scalar, scalar

    rng = np.random.default_rng(random_state)
    bootstrap_means = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sample = rng.choice(values, size=values.size, replace=True)
        bootstrap_means[index] = float(sample.mean())

    alpha = 1.0 - confidence_level
    lower = float(np.quantile(bootstrap_means, alpha / 2))
    upper = float(np.quantile(bootstrap_means, 1.0 - (alpha / 2)))
    return lower, upper


def extract_claims(text: str) -> list[str]:
    if not text.strip():
        return []
    stripped_text = DEFAULT_PROVENANCE_PATTERN.sub("", text)
    fragments = DEFAULT_CLAIM_SPLIT_PATTERN.split(stripped_text.strip())
    claims = []
    for fragment in fragments:
        normalized_fragment = re.sub(r"\s+", " ", fragment.strip()).strip(" .")
        if normalized_fragment:
            claims.append(normalized_fragment)
    return claims


def _context_to_chunks(context: str | Sequence[str]) -> list[str]:
    if isinstance(context, str):
        return [context] if context.strip() else []
    chunks: list[str] = []
    for item in context:
        if isinstance(item, str):
            normalized = item.strip()
            if normalized:
                chunks.append(normalized)
        elif isinstance(item, dict):
            for value in item.values():
                if isinstance(value, str) and value.strip():
                    chunks.append(value.strip())
    return chunks


def _normalize_answer_text(text: str) -> str:
    text = DEFAULT_PROVENANCE_PATTERN.sub("", text)
    text = re.sub(r"\b(?:direct answer|reasoning|confidence level)\s*:\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
    text = re.sub(r"[^\w\s.%/-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _canonical_label(label: str) -> str:
    normalized = _normalize_answer_text(label)
    first_token = normalized.split(" ", 1)[0] if normalized else normalized
    for canonical_label, synonyms in DEFAULT_LABEL_SYNONYMS.items():
        if normalized in synonyms or first_token in synonyms:
            return canonical_label
    return normalized


def _build_abbreviation_dictionary(texts: Sequence[str]) -> dict[str, str]:
    abbreviation_dict = dict(DEFAULT_UMLS_EXPANSIONS)
    for text in texts:
        for match in DEFAULT_ABBREVIATION_PATTERN.finditer(text):
            short_form = match.group("short").strip().lower()
            long_form = re.sub(r"\s+", " ", match.group("long").strip().lower())
            if short_form and long_form:
                abbreviation_dict[short_form] = long_form
    return abbreviation_dict


def _normalize_medical_text(text: str, abbreviation_dict: dict[str, str]) -> str:
    normalized = _normalize_answer_text(text)
    for abbreviation, expansion in abbreviation_dict.items():
        normalized = re.sub(rf"\b{re.escape(abbreviation.lower())}\b", expansion.lower(), normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _claim_supported(claim: str, context_chunks: Sequence[str], nli_model: Any | None) -> bool:
    if not claim.strip():
        return False
    claim_tokens = _tokenize(claim)
    if not claim_tokens:
        return False

    best_support_score = 0.0
    for context in context_chunks:
        if not _numeric_compatible(claim, context):
            continue
        if _claim_contradicted(claim, context, nli_model):
            continue
        overlap_score = _lexical_overlap(claim_tokens, _tokenize(context))
        entailment_score = _entailment_score(claim, context, nli_model)
        support_score = max(overlap_score, entailment_score)
        if support_score >= DEFAULT_LEXICAL_SUPPORT_THRESHOLD or entailment_score >= DEFAULT_ENTAILMENT_THRESHOLD:
            return True
        best_support_score = max(best_support_score, support_score)
    return best_support_score >= DEFAULT_LEXICAL_SUPPORT_THRESHOLD


def _claim_contradicted(claim: str, context: str, nli_model: Any | None) -> bool:
    contradiction_score = _contradiction_score(claim, context, nli_model)
    if contradiction_score >= DEFAULT_CONTRADICTION_THRESHOLD:
        return True
    return _heuristic_polarity_conflict(claim, context)


def _pairwise_contradiction(first_claim: str, second_claim: str, nli_model: Any | None) -> bool:
    if not _topic_overlap(first_claim, second_claim):
        return False
    contradiction_score = max(
        _contradiction_score(first_claim, second_claim, nli_model),
        _contradiction_score(second_claim, first_claim, nli_model),
    )
    if contradiction_score >= DEFAULT_CONTRADICTION_THRESHOLD:
        return True
    if _heuristic_polarity_conflict(first_claim, second_claim):
        return True
    return _numeric_conflict(first_claim, second_claim)


def _entailment_score(claim: str, context: str, nli_model: Any | None) -> float:
    output = _call_nli_model(context, claim, nli_model)
    if output is not None and output["label"] in {"entailment", "supported"}:
        return float(output["score"])
    if _topic_overlap(claim, context):
        return _lexical_overlap(_tokenize(claim), _tokenize(context))
    return 0.0


def _contradiction_score(claim: str, context: str, nli_model: Any | None) -> float:
    output = _call_nli_model(context, claim, nli_model)
    if output is not None and output["label"] == "contradiction":
        return float(output["score"])
    if _heuristic_polarity_conflict(claim, context):
        return 0.8
    if _numeric_conflict(claim, context):
        return 0.7
    return 0.0


def _call_nli_model(premise: str, hypothesis: str, nli_model: Any | None) -> dict[str, Any] | None:
    if nli_model is None:
        return None
    try:
        if callable(nli_model):
            output = nli_model(premise, hypothesis)
        elif hasattr(nli_model, "predict"):
            output = nli_model.predict([(premise, hypothesis)])
        else:
            return None
    except Exception as exc:  # pragma: no cover - optional runtime path
        logger.warning(f"NLI model invocation failed during evaluation: {exc}")
        return None
    return _standardize_nli_output(output)


def _standardize_nli_output(output: Any) -> dict[str, Any] | None:
    if output is None:
        return None
    if isinstance(output, list) and output:
        return _standardize_nli_output(output[0])
    if isinstance(output, dict):
        label = str(output.get("label", output.get("verdict", ""))).strip().lower()
        score = float(output.get("score", output.get("confidence", 0.0)))
        if "entail" in label or "support" in label:
            return {"label": "entailment", "score": score}
        if "contrad" in label:
            return {"label": "contradiction", "score": score}
        if "neutral" in label:
            return {"label": "neutral", "score": score}
        return {"label": label or "neutral", "score": score}
    if isinstance(output, tuple) and len(output) >= 2:
        label = str(output[0]).strip().lower()
        score = float(output[1])
        return _standardize_nli_output({"label": label, "score": score})
    if isinstance(output, (int, float)):
        score = float(output)
        label = "entailment" if score >= DEFAULT_ENTAILMENT_THRESHOLD else "neutral"
        return {"label": label, "score": score}
    return None


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if token not in DEFAULT_STOPWORDS and len(token) > 1}


def _lexical_overlap(first_tokens: set[str], second_tokens: set[str]) -> float:
    if not first_tokens or not second_tokens:
        return 0.0
    return _safe_divide(len(first_tokens & second_tokens), len(first_tokens))


def _topic_overlap(first_text: str, second_text: str) -> bool:
    first_tokens = _tokenize(first_text)
    second_tokens = _tokenize(second_text)
    return bool(first_tokens & second_tokens)


def _extract_numeric_mentions(text: str) -> list[tuple[float, str]]:
    numeric_mentions: list[tuple[float, str]] = []
    for match in DEFAULT_NUMERIC_PATTERN.finditer(text):
        value = float(match.group("value"))
        unit = (match.group("unit") or "").strip().lower()
        numeric_mentions.append((value, unit))
    return numeric_mentions


def _numeric_compatible(claim: str, context: str) -> bool:
    claim_mentions = _extract_numeric_mentions(claim)
    if not claim_mentions:
        return True
    context_mentions = _extract_numeric_mentions(context)
    if not context_mentions:
        return False

    for claim_value, claim_unit in claim_mentions:
        compatible = False
        for context_value, context_unit in context_mentions:
            if claim_unit and context_unit and claim_unit != context_unit:
                continue
            if math.isclose(claim_value, context_value, rel_tol=DEFAULT_NUMERIC_REL_TOL, abs_tol=DEFAULT_NUMERIC_ABS_TOL):
                compatible = True
                break
        if not compatible:
            return False
    return True


def _numeric_conflict(first_text: str, second_text: str) -> bool:
    first_mentions = _extract_numeric_mentions(first_text)
    second_mentions = _extract_numeric_mentions(second_text)
    if not first_mentions or not second_mentions:
        return False

    for first_value, first_unit in first_mentions:
        for second_value, second_unit in second_mentions:
            if first_unit and second_unit and first_unit != second_unit:
                continue
            if not math.isclose(first_value, second_value, rel_tol=DEFAULT_NUMERIC_REL_TOL, abs_tol=DEFAULT_NUMERIC_ABS_TOL):
                return True
    return False


def _heuristic_polarity_conflict(first_text: str, second_text: str) -> bool:
    if not _topic_overlap(first_text, second_text):
        return False

    first_lower = first_text.lower()
    second_lower = second_text.lower()
    first_positive = any(cue in first_lower for cue in DEFAULT_POSITIVE_CUES)
    first_negative = any(cue in first_lower for cue in DEFAULT_NEGATIVE_CUES)
    second_positive = any(cue in second_lower for cue in DEFAULT_POSITIVE_CUES)
    second_negative = any(cue in second_lower for cue in DEFAULT_NEGATIVE_CUES)
    return (first_positive and second_negative) or (first_negative and second_positive)


def _normalize_edge_identifier(edge: str) -> str:
    normalized = str(edge).strip()
    if normalized.startswith("edge:"):
        return normalized[len("edge:") :]
    if normalized.startswith("edge_id:"):
        return normalized[len("edge_id:") :]
    return normalized


def _normalize_claim_identifier(claim: str) -> str:
    return _normalize_answer_text(claim)


def _ensure_sequence(value: int | Sequence[str]) -> Sequence[str]:
    if isinstance(value, int):
        return [f"claim_{index}" for index in range(value)]
    return value


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _f1_from_precision_recall(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return float((2 * precision * recall) / (precision + recall))


__all__ = [
    "bootstrap_ci",
    "contradiction_detection_rate",
    "exact_match",
    "extract_claims",
    "f1_hallucination",
    "macro_f1",
    "mcnemar_test",
    "ragas_faithfulness",
    "reasoning_completeness_score",
    "reasoning_necessity_score",
]
