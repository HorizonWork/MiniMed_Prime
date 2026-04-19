from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.eval.metrics import (
    contradiction_detection_rate,
    exact_match,
    extract_claims,
    f1_hallucination,
    macro_f1,
    ragas_faithfulness,
    reasoning_completeness_score,
    reasoning_necessity_score,
)
from src.orchestrator import MedicalReasoningSystemV3, SystemConfig
from src.schemas import AnswerWithTrace
from src.utils.kaggle_env import KaggleEnv
from src.utils.structured_logger import DEFAULT_LOG_DIR, StructuredLogger

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())


DEFAULT_DATA_DIR = KaggleEnv.path("data")
DEFAULT_EVAL_DIR = DEFAULT_DATA_DIR / "eval"
DEFAULT_SUPPORTED_DATASETS = {"medqa", "medmcqa", "pubmedqa", "bioasq"}
DEFAULT_DATASET_EXTENSIONS = (".jsonl", ".json", ".csv")


@dataclass(slots=True)
class EvalExample:
    question: str
    gold_answer: str
    context: list[str] = field(default_factory=list)
    gold_edges: list[str] = field(default_factory=list)
    gold_claims: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Evaluator:
    def __init__(self, pipeline: MedicalReasoningSystemV3, dataset: str) -> None:
        self.pipeline = pipeline
        self.dataset = dataset
        self.dataset_name = Path(dataset).stem.lower()
        self.event_logger = StructuredLogger("evaluator", DEFAULT_LOG_DIR)

    def run(self, max_samples: int | None = None) -> dict[str, float]:
        examples = self._load_examples()
        if max_samples is not None:
            examples = examples[:max_samples]
        if not examples:
            return {
                "samples": 0.0,
                "exact_match": 0.0,
                "macro_f1": 0.0,
                "ragas_faithfulness": 0.0,
                "hallucination_precision": 0.0,
                "hallucination_recall": 0.0,
                "hallucination_f1": 0.0,
                "contradiction_detection_rate": 0.0,
                "reasoning_completeness_score": 0.0,
                "reasoning_necessity_score": 0.0,
                "abstention_rate": 0.0,
                "mean_confidence": 0.0,
            }

        exact_match_scores: list[float] = []
        classification_predictions: list[str] = []
        classification_golds: list[str] = []
        faithfulness_scores: list[float] = []
        contradiction_scores: list[float] = []
        completeness_scores: list[float] = []
        necessity_scores: list[float] = []
        abstentions = 0
        confidences: list[float] = []
        latency_scores: list[float] = []
        provenance_counts: list[float] = []
        answer_lengths: list[float] = []
        supported_claims_aggregate: list[str] = []
        predicted_claims_aggregate: list[str] = []
        gold_claims_aggregate: list[str] = []

        for example_index, example in enumerate(examples):
            sample_started_at = time.perf_counter()
            output = self._run_example(example)
            sample_latency_ms = (time.perf_counter() - sample_started_at) * 1000.0
            predicted_answer = self._prediction_text(output)
            gold_answer = example.gold_answer

            exact_score = exact_match(predicted_answer, gold_answer)
            exact_match_scores.append(exact_score)
            predicted_label = self._canonical_decision(predicted_answer)
            gold_label = self._canonical_decision(gold_answer)
            if predicted_label is not None and gold_label is not None:
                classification_predictions.append(predicted_label)
                classification_golds.append(gold_label)

            faithfulness = ragas_faithfulness(predicted_answer, example.context, nli_model=None)
            faithfulness_scores.append(faithfulness)
            contradiction_rate = contradiction_detection_rate(extract_claims(predicted_answer), nli_model=None)
            contradiction_scores.append(contradiction_rate)
            predicted_edges = [reference for reference in output.provenance if reference.startswith("edge:")]
            completeness_score = reasoning_completeness_score(predicted_edges, example.gold_edges)
            necessity_score = reasoning_necessity_score(predicted_edges, example.gold_edges)
            completeness_scores.append(completeness_score)
            necessity_scores.append(necessity_score)
            abstentions += int(output.abstention)
            confidences.append(output.confidence)
            latency_scores.append(sample_latency_ms)
            provenance_counts.append(float(len(output.provenance)))
            answer_lengths.append(float(len(output.answer_text)))

            predicted_claims = extract_claims(predicted_answer)
            gold_claims = example.gold_claims or extract_claims(gold_answer)
            supported_count = int(round(faithfulness * len(predicted_claims)))
            supported_claims = predicted_claims[:supported_count]
            supported_claims_aggregate.extend(supported_claims)
            predicted_claims_aggregate.extend(predicted_claims)
            gold_claims_aggregate.extend(gold_claims)
            sample_precision, sample_recall, sample_hallucination_f1 = f1_hallucination(
                claims_supported=supported_claims,
                total_claims=predicted_claims,
                gold_claims=gold_claims,
            )
            self.event_logger.log_event(
                "evaluation_sample",
                {
                    "dataset": self.dataset,
                    "sample_index": example_index,
                    "question": example.question,
                    "question_id": output.question_id,
                    "exact_match": exact_score,
                    "faithfulness": faithfulness,
                    "hallucination_precision": sample_precision,
                    "hallucination_recall": sample_recall,
                    "hallucination_f1": sample_hallucination_f1,
                    "contradiction_detection_rate": contradiction_rate,
                    "reasoning_completeness_score": completeness_score,
                    "reasoning_necessity_score": necessity_score,
                    "abstention": output.abstention,
                    "confidence": output.confidence,
                    "provenance_count": len(output.provenance),
                    "answer_char_length": len(output.answer_text),
                    "backend_used": "medical_reasoning_v3",
                    "latency_ms": sample_latency_ms,
                },
            )

        hallucination_precision, hallucination_recall, hallucination_f1 = f1_hallucination(
            claims_supported=supported_claims_aggregate,
            total_claims=predicted_claims_aggregate,
            gold_claims=gold_claims_aggregate,
        )

        results = {
            "samples": float(len(examples)),
            "exact_match": self._mean(exact_match_scores),
            "macro_f1": macro_f1(classification_predictions, classification_golds) if classification_predictions else 0.0,
            "ragas_faithfulness": self._mean(faithfulness_scores),
            "hallucination_precision": hallucination_precision,
            "hallucination_recall": hallucination_recall,
            "hallucination_f1": hallucination_f1,
            "contradiction_detection_rate": self._mean(contradiction_scores),
            "reasoning_completeness_score": self._mean(completeness_scores),
            "reasoning_necessity_score": self._mean(necessity_scores),
            "abstention_rate": abstentions / len(examples),
            "mean_confidence": self._mean(confidences),
            "latency_p50_ms": self._percentile(latency_scores, 50.0),
            "latency_p95_ms": self._percentile(latency_scores, 95.0),
            "latency_p99_ms": self._percentile(latency_scores, 99.0),
            "mean_provenance_count": self._mean(provenance_counts),
            "mean_answer_char_length": self._mean(answer_lengths),
        }
        self.event_logger.log_event(
            "evaluation_complete",
            {
                "dataset": self.dataset,
                "samples": len(examples),
                "backend_used": "medical_reasoning_v3",
                "latency_ms": 0.0,
                "metrics": results,
            },
        )
        return {metric_name: float(metric_value) for metric_name, metric_value in results.items()}

    def _run_example(self, example: EvalExample) -> AnswerWithTrace:
        return self.pipeline.answer(example.question)

    def _load_examples(self) -> list[EvalExample]:
        dataset_path = self._resolve_dataset_path()
        records = self._load_records(dataset_path)
        return [self._normalize_record(record) for record in records if self._record_has_question(record)]

    def _resolve_dataset_path(self) -> Path:
        provided_path = Path(self.dataset)
        if provided_path.exists():
            return provided_path

        candidate_paths = [
            DEFAULT_EVAL_DIR / self.dataset,
            DEFAULT_EVAL_DIR / f"{self.dataset}.jsonl",
            DEFAULT_EVAL_DIR / f"{self.dataset}.json",
            DEFAULT_EVAL_DIR / f"{self.dataset}.csv",
            DEFAULT_DATA_DIR / self.dataset,
            DEFAULT_DATA_DIR / f"{self.dataset}.jsonl",
            DEFAULT_DATA_DIR / f"{self.dataset}.json",
            DEFAULT_DATA_DIR / f"{self.dataset}.csv",
        ]
        for candidate_path in candidate_paths:
            if candidate_path.exists():
                return candidate_path

        resolved_name = self.dataset.lower()
        if resolved_name not in DEFAULT_SUPPORTED_DATASETS:
            raise FileNotFoundError(f"Dataset '{self.dataset}' was not found locally.")
        raise FileNotFoundError(
            f"Dataset '{self.dataset}' was not found. Expected a local JSONL/JSON/CSV file under {DEFAULT_DATA_DIR} or {DEFAULT_EVAL_DIR}."
        )

    def _load_records(self, dataset_path: Path) -> list[dict[str, Any]]:
        if dataset_path.is_dir():
            for extension in DEFAULT_DATASET_EXTENSIONS:
                matches = sorted(dataset_path.glob(f"*{extension}"))
                if matches:
                    return self._load_records(matches[0])
            raise FileNotFoundError(f"No supported dataset file found under directory {dataset_path}.")

        suffix = dataset_path.suffix.lower()
        if suffix == ".jsonl":
            return self._load_jsonl_records(dataset_path)
        if suffix == ".json":
            return self._load_json_records(dataset_path)
        if suffix == ".csv":
            return self._load_csv_records(dataset_path)
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")

    def _load_jsonl_records(self, dataset_path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if isinstance(payload, dict):
                    records.append(payload)
        return records

    def _load_json_records(self, dataset_path: Path) -> list[dict[str, Any]]:
        with dataset_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return [record for record in payload if isinstance(record, dict)]
        if isinstance(payload, dict):
            for key in ("data", "records", "examples", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [record for record in value if isinstance(record, dict)]
            for split_name in ("validation", "dev", "test", "train"):
                value = payload.get(split_name)
                if isinstance(value, list):
                    return [record for record in value if isinstance(record, dict)]
        raise ValueError(f"Could not interpret JSON dataset structure from {dataset_path}.")

    def _load_csv_records(self, dataset_path: Path) -> list[dict[str, Any]]:
        with dataset_path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    def _normalize_record(self, record: dict[str, Any]) -> EvalExample:
        question = self._string_field(record, "question", "question_text", "query", "body", "input")
        gold_answer = self._extract_gold_answer(record)
        context = self._extract_context(record)
        gold_edges = self._extract_string_list(record, "gold_edges", "supporting_edges", "reasoning_edges", "edge_ids")
        gold_claims = self._extract_gold_claims(record, gold_answer)
        metadata = {key: value for key, value in record.items() if key not in {"question", "question_text", "query", "body", "input"}}
        return EvalExample(
            question=question,
            gold_answer=gold_answer,
            context=context,
            gold_edges=gold_edges,
            gold_claims=gold_claims,
            metadata=metadata,
        )

    def _extract_gold_answer(self, record: dict[str, Any]) -> str:
        dataset_name = self.dataset_name.lower()
        if dataset_name == "pubmedqa":
            return self._string_field(record, "final_decision", "answer", "label", default="")
        if dataset_name in {"medqa", "medmcqa"}:
            answer_index = record.get("answer_idx", record.get("cop", record.get("correct_option")))
            options = record.get("options", record.get("choices"))
            if answer_index is not None and options is not None:
                resolved_option = self._resolve_option_answer(answer_index, options)
                if resolved_option:
                    return resolved_option
            return self._string_field(record, "answer", "label", "output", default="")
        if dataset_name == "bioasq":
            ideal_answer = record.get("ideal_answer")
            if isinstance(ideal_answer, list):
                return " ".join(str(item).strip() for item in ideal_answer if str(item).strip())
        return self._string_field(record, "answer", "gold", "label", "output", "ideal_answer", default="")

    def _extract_context(self, record: dict[str, Any]) -> list[str]:
        context_values = []
        for key in ("context", "contexts", "abstract", "abstracts", "explanation", "supporting_facts", "snippets", "documents"):
            value = record.get(key)
            if value is None:
                continue
            context_values.extend(self._flatten_text_values(value))
        return context_values

    def _extract_gold_claims(self, record: dict[str, Any], gold_answer: str) -> list[str]:
        explicit_claims = self._extract_string_list(record, "gold_claims", "supporting_claims")
        if explicit_claims:
            return explicit_claims
        if gold_answer:
            claims = extract_claims(gold_answer)
            if claims:
                return claims
        return []

    def _extract_string_list(self, record: dict[str, Any], *keys: str) -> list[str]:
        extracted: list[str] = []
        for key in keys:
            value = record.get(key)
            if value is None:
                continue
            extracted.extend(self._flatten_text_values(value))
        deduplicated = []
        seen = set()
        for item in extracted:
            if item in seen:
                continue
            seen.add(item)
            deduplicated.append(item)
        return deduplicated

    def _resolve_option_answer(self, answer_index: Any, options: Any) -> str:
        if isinstance(options, dict):
            normalized_index = str(answer_index).strip()
            for key in (normalized_index, normalized_index.lower(), normalized_index.upper()):
                if key in options:
                    return str(options[key]).strip()
        if isinstance(options, list):
            if isinstance(answer_index, str):
                option_letters = "abcdefghijklmnopqrstuvwxyz"
                lowered = answer_index.strip().lower()
                if lowered in option_letters:
                    index = option_letters.index(lowered)
                else:
                    index = int(lowered)
            else:
                index = int(answer_index)
            if 0 <= index < len(options):
                return str(options[index]).strip()
        return ""

    def _flatten_text_values(self, value: Any) -> list[str]:
        if isinstance(value, str):
            normalized = value.strip()
            return [normalized] if normalized else []
        if isinstance(value, list):
            flattened: list[str] = []
            for item in value:
                flattened.extend(self._flatten_text_values(item))
            return flattened
        if isinstance(value, dict):
            flattened: list[str] = []
            for item in value.values():
                flattened.extend(self._flatten_text_values(item))
            return flattened
        return [str(value).strip()] if str(value).strip() else []

    def _string_field(self, record: dict[str, Any], *keys: str, default: str = "") -> str:
        for key in keys:
            value = record.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                normalized_list = [str(item).strip() for item in value if str(item).strip()]
                if normalized_list:
                    return " ".join(normalized_list)
            normalized = str(value).strip()
            if normalized:
                return normalized
        return default

    def _record_has_question(self, record: dict[str, Any]) -> bool:
        return bool(self._string_field(record, "question", "question_text", "query", "body", "input"))

    def _prediction_text(self, output: AnswerWithTrace) -> str:
        if self._canonical_decision(output.answer_text) is not None:
            return self._canonical_decision(output.answer_text) or output.answer_text
        return output.answer_text

    @staticmethod
    def _canonical_decision(text: str) -> str | None:
        normalized = text.lower()
        for label in ("yes", "no", "maybe"):
            if re.search(rf"\b{label}\b", normalized):
                return label
        return None

    @staticmethod
    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        if len(sorted_values) == 1:
            return float(sorted_values[0])
        rank = (len(sorted_values) - 1) * (percentile / 100.0)
        lower_index = int(rank)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)
        weight = rank - lower_index
        return float(sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MiniMed Prime evaluation.")
    parser.add_argument("dataset", help="Dataset name or local dataset path.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on the number of evaluated samples.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write aggregate metrics JSON.")
    parser.add_argument("--kaggle", action="store_true", help="Resolve output paths with KaggleEnv.")
    args = parser.parse_args()

    pipeline = MedicalReasoningSystemV3(SystemConfig())
    evaluator = Evaluator(pipeline=pipeline, dataset=args.dataset)
    results = evaluator.run(max_samples=args.max_samples)
    if args.output_json is not None:
        output_path = KaggleEnv.ensure_writeable(KaggleEnv.path(args.output_json) if args.kaggle else args.output_json)
        output_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(f"Evaluation results: {json.dumps(results, sort_keys=True)}")


if __name__ == "__main__":
    main()
