#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


try:  # pragma: no cover - exercised indirectly in integration flow
    from src.utils.kaggle_env import KaggleEnv, T4Hardening
except Exception:  # pragma: no cover - fallback for import failures
    class KaggleEnv:  # type: ignore[no-redef]
        @staticmethod
        def is_kaggle() -> bool:
            return False

        @staticmethod
        def path(path: str | Path) -> Path:
            return Path(path)

        @staticmethod
        def ensure_writeable(path: Path) -> Path:
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

    class T4Hardening:  # type: ignore[no-redef]
        @staticmethod
        def setup_memory() -> None:
            return None


try:  # pragma: no cover - import availability varies by environment
    from src.layers.layer1_retrieval import (
        DEFAULT_PRIMEKG_TOP_K,
        EntityLinker,
        PrimeKGExtractor,
        QUESTION_TYPE_PATTERNS,
        QUESTION_TYPE_RELATION_FILTERS,
    )
except Exception:  # pragma: no cover - fallback path is covered by unit tests
    DEFAULT_PRIMEKG_TOP_K = 200
    EntityLinker = None  # type: ignore[assignment]
    PrimeKGExtractor = None  # type: ignore[assignment]
    QUESTION_TYPE_PATTERNS = {
        "drug_interaction": ("interaction", "interact", "contraindicated", "combine"),
        "dosage": ("dose", "dosage", "mg", "mcg"),
        "diagnosis": ("diagnosis", "diagnose", "differential", "most likely"),
        "etiology": ("cause", "causes", "etiology", "mechanism"),
        "factoid": (),
        "other": (),
    }
    QUESTION_TYPE_RELATION_FILTERS = {
        "drug_interaction": {"drug_drug", "contraindication", "drug_effect", "drug_protein"},
        "etiology": {"disease_disease", "disease_protein", "exposure_disease", "exposure_protein", "phenotype_protein"},
        "diagnosis": {"disease_phenotype_negative", "disease_phenotype_positive", "phenotype_phenotype", "disease_disease"},
        "dosage": {"drug_protein", "drug_effect", "contraindication"},
        "factoid": set(),
        "other": set(),
    }

try:  # pragma: no cover - import availability varies by environment
    from src.layers.layer2_embedder import MedicalGraphEmbedder
except Exception:  # pragma: no cover
    MedicalGraphEmbedder = None  # type: ignore[assignment]

try:  # pragma: no cover
    from src.schemas import EvidenceBundle, QuestionEntity
except Exception:  # pragma: no cover
    EvidenceBundle = None  # type: ignore[assignment]
    QuestionEntity = None  # type: ignore[assignment]

try:  # pragma: no cover
    from src.training.medreason_adapter import map_medreason_edges, normalize_medreason_record
except Exception:  # pragma: no cover
    map_medreason_edges = None  # type: ignore[assignment]
    normalize_medreason_record = None  # type: ignore[assignment]

try:  # pragma: no cover
    from src.training.trm_dataset_builder import path_to_token_sequence
except Exception:  # pragma: no cover
    path_to_token_sequence = None  # type: ignore[assignment]


QUESTION_KEYS = (
    "question",
    "question_text",
    "query",
    "prompt",
    "problem",
    "input",
    "medical_question",
)
ANSWER_KEYS = (
    "answer",
    "final_answer",
    "gold_answer",
    "output",
    "response",
    "label",
)
REASONING_KEYS = (
    "reasoning",
    "cot",
    "chain_of_thought",
    "rationale",
    "explanation",
    "steps",
    "kg_cot",
    "knowledge_graph",
    "path",
)
VALID_BUCKETS = ("stable_graph", "threshold_a", "threshold_b", "hard_absent")
VALID_PASSES = ("exact", "normalized", "broad")
CSV_HEADERS = (
    "sample_id",
    "row_index",
    "dataset_name",
    "id_in_dataset",
    "question_type",
    "pass_success_count",
    "support_rate",
    "max_node_count",
    "max_edge_count",
    "max_labeled_token_count",
    "bucket",
    "bucket_reason",
    "error_count",
)
EDGE_ID_PATTERN = re.compile(
    r"(?:edge_id|edge)\s*[:=]\s*['\"]?([A-Za-z0-9_.:-]+)|\[edge:([^\]]+)\]",
    re.IGNORECASE,
)
MCQ_HEADER_PATTERN = re.compile(r"^\s*(?:answer\s+choices?|choices?|options?)\s*:?\s*$", re.IGNORECASE)
MCQ_PREFIX_PATTERN = re.compile(r"^\s*(?:[\(\[]?[A-Z][\)\].:-]|[\(\[]?\d+[\)\].:-]|[-*])\s*")
PUNCT_CLEAN_PATTERN = re.compile(r"[^a-z0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")
ABBREVIATION_MAP = {
    "t1dm": "type 1 diabetes",
    "type i diabetes": "type 1 diabetes",
    "t2dm": "type 2 diabetes",
    "type ii diabetes": "type 2 diabetes",
    "af": "atrial fibrillation",
    "mi": "myocardial infarction",
    "htn": "hypertension",
    "copd": "chronic obstructive pulmonary disease",
    "uti": "urinary tract infection",
}
BROAD_TOKEN_CAP = 128
BROAD_TOP_K_CAP = 400
CANDIDATE_PATH_CAP = 50_000
PROGRESS_PRINT_EVERY = 25


@dataclass(slots=True)
class SimpleEntity:
    surface: str
    cui: str | None = None
    primekg_node_id: str | None = None
    entity_type: str = "other"


@dataclass(slots=True)
class LocalExample:
    question: str
    answer: str
    reasoning: Any
    direct_edge_ids: list[str]
    group_id: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PassResult:
    pass_name: str
    success: bool
    entity_count: int
    linked_entity_count: int
    node_count: int
    edge_count: int
    candidate_path_count: int
    labeled_token_count: int
    error_message: str | None
    duration_ms: float

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["duration_ms"] = round(float(self.duration_ms), 3)
        return payload


@dataclass(slots=True)
class SampleTriageResult:
    sample_id: str
    row_index: int
    question: str
    question_type: str
    pass_success_count: int
    support_rate: float
    max_node_count: int
    max_edge_count: int
    max_labeled_token_count: int
    bucket: str
    bucket_reason: str
    passes: list[PassResult] = field(default_factory=list)
    dataset_name: str | None = None
    id_in_dataset: Any = None
    total_passes: int = 0
    error_count: int = 0
    source: str | None = None
    split: str | None = None
    selected_passes: list[str] = field(default_factory=list)
    fallback_warnings: list[str] = field(default_factory=list)
    run_config: dict[str, Any] = field(default_factory=dict)
    sample_duration_ms: float = 0.0

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["support_rate"] = round(float(self.support_rate), 6)
        payload["sample_duration_ms"] = round(float(self.sample_duration_ms), 3)
        payload["passes"] = [item.to_json_dict() for item in self.passes]
        return payload


@dataclass(slots=True)
class RuntimeDependencies:
    linker: Any
    kg_extractor: Any
    embedder: Any
    warnings: list[str]
    default_top_k: int
    linker_lock: Lock = field(default_factory=Lock)
    embedder_lock: Lock = field(default_factory=Lock)
    token_index_lock: Lock = field(default_factory=Lock)
    broad_match_cache_lock: Lock = field(default_factory=Lock)
    node_token_index: dict[str, list[str]] | None = None
    broad_match_cache: dict[str, str | None] = field(default_factory=dict)


@dataclass(slots=True)
class PersistedState:
    started_at: str
    updated_at: str
    total_target: int
    total_done: int = 0
    total_errors: int = 0
    last_row_index: int = -1
    bucket_counts: dict[str, int] = field(default_factory=lambda: {bucket: 0 for bucket in VALID_BUCKETS})
    duration_sum_ms: float = 0.0
    skip_row_indexes: set[int] = field(default_factory=set)
    skip_sample_ids: set[str] = field(default_factory=set)
    run_config: dict[str, Any] = field(default_factory=dict)

    @property
    def total_seen(self) -> int:
        return self.total_done

    @property
    def avg_ms(self) -> float:
        return self.duration_sum_ms / self.total_done if self.total_done else 0.0


class RegexFallbackEntityLinker:
    backend_used = "fallback_regex"

    def link(self, text: str) -> list[SimpleEntity]:
        candidates: list[SimpleEntity] = []
        seen: set[str] = set()
        for phrase in _iter_candidate_phrases(text):
            normalized = _normalize_name(phrase)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(SimpleEntity(surface=phrase, entity_type=_infer_entity_type(phrase)))
        return candidates


class EmptyGraphExtractor:
    backend_used = "empty_graph"

    def __init__(self) -> None:
        self.graph: Any = None
        self.node_name_index: dict[str, set[str]] = {}
        self.node_degrees: dict[str, int] = {}

    def resolve_seed_entities(self, seed_entities: Sequence[Any]) -> list[Any]:
        return [_copy_entity(entity, primekg_node_id=_entity_node_id(entity)) for entity in seed_entities]

    def extract_2hop(
        self,
        seed_entities: Sequence[Any],
        top_k: int = DEFAULT_PRIMEKG_TOP_K,
        relation_filter: set[str] | None = None,
    ) -> list[Any]:
        del seed_entities, top_k, relation_filter
        return []


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Triage MedReason-style JSONL samples by graph-groundability before TRM training.")
    parser.add_argument("--input-jsonl", type=Path, default=None, help="Input MedReason-style JSONL path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for triage outputs.")
    parser.add_argument("--source", default="data/medreason", help="Metadata only when --input-jsonl is provided.")
    parser.add_argument("--split", default="train", help="Metadata only when --input-jsonl is provided.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--passes", default="exact,normalized,broad", help="Comma-separated pass list.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--kaggle", action="store_true")
    parser.add_argument("--dry-run", type=int, default=None, help="Run only the first N selected samples.")
    parser.add_argument("--validate-only", action="store_true", help="Validate existing outputs and exit.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    output_dir = KaggleEnv.ensure_writeable(args.output_dir if args.output_dir.is_absolute() else KaggleEnv.path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)

    if args.kaggle or KaggleEnv.is_kaggle():
        T4Hardening.setup_memory()

    if args.validate_only:
        validation = validate_outputs(output_dir)
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0 if validation.get("valid") else 1

    if args.input_jsonl is None:
        parser.error("--input-jsonl is required unless --validate-only is set.")

    selected_passes = parse_passes(args.passes)
    if args.workers < 1:
        parser.error("--workers must be >= 1.")
    if args.start_index < 0:
        parser.error("--start-index must be >= 0.")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0.")
    if args.dry_run is not None and args.dry_run < 0:
        parser.error("--dry-run must be >= 0.")
    if args.save_every < 1:
        parser.error("--save-every must be >= 1.")

    input_jsonl = args.input_jsonl if args.input_jsonl.is_absolute() else KaggleEnv.path(args.input_jsonl)
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL was not found: {input_jsonl}")

    run_config = {
        "input_jsonl": str(input_jsonl.resolve()),
        "source": str(args.source),
        "split": str(args.split),
        "passes": list(selected_passes),
    }
    target_total = compute_target_total(input_jsonl, start_index=args.start_index, limit=args.limit, dry_run=args.dry_run)
    state = prepare_output_state(output_dir=output_dir, resume=args.resume, run_config=run_config, target_total=target_total, logger=logger)
    dependencies = build_runtime_dependencies(logger=logger)

    logger.info(
        "Starting triage: input=%s target=%s resume=%s workers=%s passes=%s",
        input_jsonl,
        state.total_target,
        args.resume,
        args.workers,
        ",".join(selected_passes),
    )

    manifest_path = output_dir / "triage_manifest.jsonl"
    csv_path = output_dir / "triage_manifest.csv"
    progress_path = output_dir / "progress.json"
    summary_path = output_dir / "triage_summary.json"
    bucket_paths = {bucket: output_dir / f"bucket_{bucket}.jsonl" for bucket in VALID_BUCKETS}

    pending_results: dict[int, Future[dict[str, Any]]] = {}
    sequence_to_commit = 0
    next_sequence = 0
    batch: list[dict[str, Any]] = []
    new_completed_since_print = 0
    max_pending = max(1, args.workers * 4)
    executor: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=args.workers) if args.workers > 1 else None

    try:
        for row_index, record in load_records(input_jsonl, start_index=args.start_index, limit=args.limit, dry_run=args.dry_run):
            sample_id = build_sample_id(record, row_index)
            if row_index in state.skip_row_indexes or sample_id in state.skip_sample_ids:
                continue
            if executor is None:
                result = process_sample(
                    row_index=row_index,
                    record=record,
                    selected_passes=selected_passes,
                    source=str(args.source),
                    split=str(args.split),
                    run_config=run_config,
                    dependencies=dependencies,
                    logger=logger,
                )
                batch.append(result)
                state = update_persisted_state(state, [result])
                new_completed_since_print += 1
            else:
                future = executor.submit(
                    process_sample,
                    row_index=row_index,
                    record=record,
                    selected_passes=selected_passes,
                    source=str(args.source),
                    split=str(args.split),
                    run_config=run_config,
                    dependencies=dependencies,
                    logger=logger,
                )
                pending_results[next_sequence] = future
                next_sequence += 1
                while len(pending_results) >= max_pending:
                    result = pending_results.pop(sequence_to_commit).result()
                    sequence_to_commit += 1
                    batch.append(result)
                    state = update_persisted_state(state, [result])
                    new_completed_since_print += 1

            if len(batch) >= args.save_every:
                persist_batch(
                    batch=batch,
                    manifest_path=manifest_path,
                    csv_path=csv_path,
                    bucket_paths=bucket_paths,
                    progress_path=progress_path,
                    summary_path=summary_path,
                    state=state,
                    logger=logger,
                )
                batch = []
            if new_completed_since_print >= PROGRESS_PRINT_EVERY:
                print_progress(state)
                new_completed_since_print = 0

        if executor is not None:
            while sequence_to_commit in pending_results:
                result = pending_results.pop(sequence_to_commit).result()
                sequence_to_commit += 1
                batch.append(result)
                state = update_persisted_state(state, [result])
                new_completed_since_print += 1
                if len(batch) >= args.save_every:
                    persist_batch(
                        batch=batch,
                        manifest_path=manifest_path,
                        csv_path=csv_path,
                        bucket_paths=bucket_paths,
                        progress_path=progress_path,
                        summary_path=summary_path,
                        state=state,
                        logger=logger,
                    )
                    batch = []
                if new_completed_since_print >= PROGRESS_PRINT_EVERY:
                    print_progress(state)
                    new_completed_since_print = 0

        if batch:
            persist_batch(
                batch=batch,
                manifest_path=manifest_path,
                csv_path=csv_path,
                bucket_paths=bucket_paths,
                progress_path=progress_path,
                summary_path=summary_path,
                state=state,
                logger=logger,
            )
        validation = validate_outputs(output_dir)
        print_progress(state)
        print(json.dumps(build_final_summary(state=state, output_dir=output_dir, validation=validation), indent=2, sort_keys=True))
        return 0 if validation.get("valid") else 1
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)


def prepare_output_state(
    *,
    output_dir: Path,
    resume: bool,
    run_config: Mapping[str, Any],
    target_total: int,
    logger: logging.Logger,
) -> PersistedState:
    manifest_path = output_dir / "triage_manifest.jsonl"
    progress_path = output_dir / "progress.json"
    started_at = read_existing_started_at(progress_path) or utc_now_iso()
    if not resume:
        if manifest_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing manifest without --resume: {manifest_path}")
        return PersistedState(
            started_at=started_at,
            updated_at=utc_now_iso(),
            total_target=target_total,
            run_config=dict(run_config),
        )

    if not manifest_path.exists():
        raise FileNotFoundError(f"Cannot resume because the manifest does not exist: {manifest_path}")

    state = reconcile_outputs_from_manifest(output_dir=output_dir, started_at=started_at, target_total=target_total, logger=logger)
    if state.run_config:
        stable_mismatch: list[str] = []
        for key in ("input_jsonl", "source", "split", "passes"):
            if state.run_config.get(key) != run_config.get(key):
                stable_mismatch.append(f"{key}: manifest={state.run_config.get(key)!r} requested={run_config.get(key)!r}")
        if stable_mismatch:
            raise ValueError("Resume configuration mismatch:\n" + "\n".join(stable_mismatch))
    else:
        state.run_config = dict(run_config)
    state.total_target = max(target_total, state.total_done)
    return state


def build_runtime_dependencies(*, logger: logging.Logger) -> RuntimeDependencies:
    warnings_list: list[str] = []
    if EntityLinker is not None:
        try:
            linker = EntityLinker()
        except Exception as exc:  # pragma: no cover - fallback covered via monkeypatch tests
            warnings_list.append(f"entity_linker_fallback:{type(exc).__name__}:{exc}")
            logger.warning("Falling back to regex entity linker because EntityLinker initialization failed: %s", exc)
            linker = RegexFallbackEntityLinker()
    else:
        warnings_list.append("entity_linker_fallback:import_failed")
        logger.warning("Falling back to regex entity linker because src.layers.layer1_retrieval could not be imported.")
        linker = RegexFallbackEntityLinker()

    if PrimeKGExtractor is not None:
        try:
            kg_extractor = PrimeKGExtractor(primekg_path=KaggleEnv.path("data/kg/primekg"))
        except Exception as exc:  # pragma: no cover - fallback covered via monkeypatch tests
            warnings_list.append(f"primekg_fallback:{type(exc).__name__}:{exc}")
            logger.warning("Falling back to empty graph extractor because PrimeKGExtractor initialization failed: %s", exc)
            kg_extractor = EmptyGraphExtractor()
    else:
        warnings_list.append("primekg_fallback:import_failed")
        logger.warning("Falling back to empty graph extractor because src.layers.layer1_retrieval could not be imported.")
        kg_extractor = EmptyGraphExtractor()

    if MedicalGraphEmbedder is not None:
        try:
            embedder = MedicalGraphEmbedder(device="cpu")
        except Exception as exc:  # pragma: no cover - fallback covered via monkeypatch tests
            warnings_list.append(f"tensorization_fallback:{type(exc).__name__}:{exc}")
            logger.warning("Disabling tensorization preview because MedicalGraphEmbedder initialization failed: %s", exc)
            embedder = None
    else:
        warnings_list.append("tensorization_fallback:embedder_import_failed")
        embedder = None

    if path_to_token_sequence is None:
        warnings_list.append("tensorization_fallback:path_to_token_sequence_import_failed")
    if map_medreason_edges is None:
        warnings_list.append("tensorization_fallback:map_medreason_edges_import_failed")

    return RuntimeDependencies(
        linker=linker,
        kg_extractor=kg_extractor,
        embedder=embedder,
        warnings=warnings_list,
        default_top_k=int(DEFAULT_PRIMEKG_TOP_K),
    )


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("triage_graph_groundability")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    log_path = output_dir / "triage.log"
    if logger.handlers:
        existing_paths = {
            getattr(handler, "baseFilename", None)
            for handler in logger.handlers
            if hasattr(handler, "baseFilename")
        }
        if str(log_path) in existing_paths:
            return logger
        logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def parse_passes(raw_value: str) -> tuple[str, ...]:
    selected = tuple(item.strip().lower() for item in raw_value.split(",") if item.strip())
    if not selected:
        raise ValueError("At least one pass must be selected.")
    invalid = sorted(set(selected) - set(VALID_PASSES))
    if invalid:
        raise ValueError(f"Unsupported passes: {', '.join(invalid)}")
    deduped: list[str] = []
    for item in selected:
        if item not in deduped:
            deduped.append(item)
    return tuple(deduped)


def compute_effective_limit(limit: int | None, dry_run: int | None) -> int | None:
    limits = [value for value in (limit, dry_run) if value is not None]
    return min(limits) if limits else None


def count_jsonl_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def compute_target_total(path: Path, *, start_index: int, limit: int | None, dry_run: int | None) -> int:
    total_records = count_jsonl_records(path)
    remaining = max(total_records - start_index, 0)
    effective_limit = compute_effective_limit(limit, dry_run)
    if effective_limit is None:
        return remaining
    return max(min(remaining, effective_limit), 0)


def load_records(
    input_jsonl: Path,
    *,
    start_index: int = 0,
    limit: int | None = None,
    dry_run: int | None = None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    emitted = 0
    effective_limit = compute_effective_limit(limit, dry_run)
    with input_jsonl.open("r", encoding="utf-8") as handle:
        row_index = -1
        for line in handle:
            if not line.strip():
                continue
            row_index += 1
            if row_index < start_index:
                continue
            if effective_limit is not None and emitted >= effective_limit:
                break
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected each JSONL row to decode to an object, got {type(record).__name__}.")
            yield row_index, record
            emitted += 1


def strip_mcq_prefixes(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if MCQ_HEADER_PATTERN.match(line):
            continue
        line = MCQ_PREFIX_PATTERN.sub("", line).strip()
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def normalize_question_text(text: str) -> str:
    normalized = strip_mcq_prefixes(text)
    lowered = normalized.lower()
    for source, target in ABBREVIATION_MAP.items():
        lowered = re.sub(rf"\b{re.escape(source)}\b", target, lowered)
    lowered = lowered.replace("&", " and ")
    lowered = PUNCT_CLEAN_PATTERN.sub(" ", lowered)
    lowered = WHITESPACE_PATTERN.sub(" ", lowered).strip()
    return lowered


def run_pass_exact(*, row_index: int, record: Mapping[str, Any], dependencies: RuntimeDependencies) -> PassResult:
    question, options = extract_question_and_options(record)
    question_text = render_question_with_options(question, options)
    return execute_pass(
        pass_name="exact",
        row_index=row_index,
        record=record,
        question_text=question_text,
        option_texts=options,
        dependencies=dependencies,
        relation_filter_mode="default",
        normalize_text=False,
        broad_mode=False,
    )


def run_pass_normalized(*, row_index: int, record: Mapping[str, Any], dependencies: RuntimeDependencies) -> PassResult:
    question, options = extract_question_and_options(record)
    normalized_question = normalize_question_text(question)
    normalized_options = [normalize_question_text(option) for option in options if normalize_question_text(option)]
    question_text = render_question_with_options(normalized_question, normalized_options)
    return execute_pass(
        pass_name="normalized",
        row_index=row_index,
        record=record,
        question_text=question_text,
        option_texts=normalized_options,
        dependencies=dependencies,
        relation_filter_mode="default",
        normalize_text=True,
        broad_mode=False,
    )


def run_pass_broad(*, row_index: int, record: Mapping[str, Any], dependencies: RuntimeDependencies) -> PassResult:
    question, options = extract_question_and_options(record)
    broadened_options = [strip_mcq_prefixes(option) for option in options if strip_mcq_prefixes(option)]
    question_text = render_question_with_options(question, broadened_options)
    return execute_pass(
        pass_name="broad",
        row_index=row_index,
        record=record,
        question_text=question_text,
        option_texts=broadened_options,
        dependencies=dependencies,
        relation_filter_mode="none",
        normalize_text=False,
        broad_mode=True,
    )


def execute_pass(
    *,
    pass_name: str,
    row_index: int,
    record: Mapping[str, Any],
    question_text: str,
    option_texts: Sequence[str],
    dependencies: RuntimeDependencies,
    relation_filter_mode: str,
    normalize_text: bool,
    broad_mode: bool,
) -> PassResult:
    started_at = time.perf_counter()
    entity_count = 0
    linked_entity_count = 0
    node_count = 0
    edge_count = 0
    candidate_path_count = 0
    labeled_token_count = 0
    error_message: str | None = None

    try:
        question_type = classify_question_type(question_text)
        relation_filter = None if relation_filter_mode == "none" else default_relation_filter(question_type)
        raw_entities = link_entities(question_text, dependencies)
        if broad_mode:
            for option in option_texts:
                raw_entities = merge_entities(raw_entities, link_entities(option, dependencies))
        entity_count = len(raw_entities)

        resolved_entities = resolve_entities(raw_entities, dependencies)
        if broad_mode:
            resolved_entities = augment_entities_broad(resolved_entities, dependencies=dependencies)
        linked_entity_count = sum(1 for entity in resolved_entities if _entity_node_id(entity))
        candidate_path_count = estimate_candidate_path_count(
            kg_extractor=dependencies.kg_extractor,
            resolved_entities=resolved_entities,
            relation_filter=relation_filter,
        )
        top_k = min(int(dependencies.default_top_k) * 2, BROAD_TOP_K_CAP) if broad_mode else int(dependencies.default_top_k)
        edges = extract_subgraph_edges(
            kg_extractor=dependencies.kg_extractor,
            resolved_entities=resolved_entities,
            top_k=top_k,
            relation_filter=relation_filter,
        )
        edge_count = len(edges)
        node_count = count_nodes(resolved_entities, edges)
        labeled_token_count = compute_labeled_token_count(
            row_index=row_index,
            record=record,
            question_text=question_text,
            question_type=question_type,
            question_entities=resolved_entities,
            subgraph_edges=edges,
            dependencies=dependencies,
            normalize_text=normalize_text,
        )
        success = node_count > 0
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        success = False
    return PassResult(
        pass_name=pass_name,
        success=success,
        entity_count=entity_count,
        linked_entity_count=linked_entity_count,
        node_count=node_count,
        edge_count=edge_count,
        candidate_path_count=min(candidate_path_count, CANDIDATE_PATH_CAP),
        labeled_token_count=labeled_token_count,
        error_message=error_message,
        duration_ms=(time.perf_counter() - started_at) * 1000.0,
    )


def process_sample(
    *,
    row_index: int,
    record: Mapping[str, Any],
    selected_passes: Sequence[str],
    source: str,
    split: str,
    run_config: Mapping[str, Any],
    dependencies: RuntimeDependencies,
    logger: logging.Logger,
) -> dict[str, Any]:
    question, _ = extract_question_and_options(record)
    sample_id = build_sample_id(record, row_index)
    question_type = classify_question_type(question or "")
    pass_results: list[PassResult] = []
    started_at = time.perf_counter()

    try:
        for pass_name in selected_passes:
            if pass_name == "exact":
                pass_results.append(run_pass_exact(row_index=row_index, record=record, dependencies=dependencies))
            elif pass_name == "normalized":
                pass_results.append(run_pass_normalized(row_index=row_index, record=record, dependencies=dependencies))
            elif pass_name == "broad":
                pass_results.append(run_pass_broad(row_index=row_index, record=record, dependencies=dependencies))
            else:  # pragma: no cover - guarded by parse_passes
                raise ValueError(f"Unsupported pass: {pass_name}")
        result = summarize_sample(
            record=record,
            row_index=row_index,
            sample_id=sample_id,
            question=question,
            question_type=question_type,
            passes=pass_results,
            source=source,
            split=split,
            run_config=run_config,
            fallback_warnings=dependencies.warnings,
            sample_duration_ms=(time.perf_counter() - started_at) * 1000.0,
        )
        return result.to_json_dict()
    except Exception as exc:
        logger.exception("Sample-level triage failure at row %s (%s).", row_index, sample_id)
        bucket, bucket_reason = assign_bucket(
            pass_success_count=sum(1 for item in pass_results if item.success),
            total_passes=len(selected_passes),
            support_rate=(sum(1 for item in pass_results if item.success) / len(selected_passes)) if selected_passes else 0.0,
            max_node_count=max((item.node_count for item in pass_results), default=0),
            max_labeled_token_count=max((item.labeled_token_count for item in pass_results), default=0),
            exception_override={
                "type": type(exc).__name__,
                "message": str(exc),
                "has_partial_graph": any(item.node_count > 0 or item.linked_entity_count > 0 for item in pass_results),
            },
        )
        result = SampleTriageResult(
            sample_id=sample_id,
            row_index=row_index,
            question=question,
            question_type=question_type,
            pass_success_count=sum(1 for item in pass_results if item.success),
            support_rate=(sum(1 for item in pass_results if item.success) / len(selected_passes)) if selected_passes else 0.0,
            max_node_count=max((item.node_count for item in pass_results), default=0),
            max_edge_count=max((item.edge_count for item in pass_results), default=0),
            max_labeled_token_count=max((item.labeled_token_count for item in pass_results), default=0),
            bucket=bucket,
            bucket_reason=bucket_reason,
            passes=pass_results,
            dataset_name=_string_or_none(record.get("dataset_name")),
            id_in_dataset=record.get("id_in_dataset"),
            total_passes=len(selected_passes),
            error_count=max(sum(1 for item in pass_results if item.error_message), 1),
            source=source,
            split=split,
            selected_passes=list(selected_passes),
            fallback_warnings=list(dependencies.warnings),
            run_config=dict(run_config),
            sample_duration_ms=(time.perf_counter() - started_at) * 1000.0,
        )
        return result.to_json_dict()


def summarize_sample(
    *,
    record: Mapping[str, Any],
    row_index: int,
    sample_id: str,
    question: str,
    question_type: str,
    passes: Sequence[PassResult],
    source: str,
    split: str,
    run_config: Mapping[str, Any],
    fallback_warnings: Sequence[str],
    sample_duration_ms: float,
) -> SampleTriageResult:
    total_passes = len(passes)
    pass_success_count = sum(1 for item in passes if item.success)
    support_rate = pass_success_count / total_passes if total_passes else 0.0
    max_node_count = max((item.node_count for item in passes), default=0)
    max_edge_count = max((item.edge_count for item in passes), default=0)
    max_labeled_token_count = max((item.labeled_token_count for item in passes), default=0)
    bucket, bucket_reason = assign_bucket(
        pass_success_count=pass_success_count,
        total_passes=total_passes,
        support_rate=support_rate,
        max_node_count=max_node_count,
        max_labeled_token_count=max_labeled_token_count,
        exception_override=None,
    )
    return SampleTriageResult(
        sample_id=sample_id,
        row_index=row_index,
        question=question,
        question_type=question_type,
        pass_success_count=pass_success_count,
        support_rate=support_rate,
        max_node_count=max_node_count,
        max_edge_count=max_edge_count,
        max_labeled_token_count=max_labeled_token_count,
        bucket=bucket,
        bucket_reason=bucket_reason,
        passes=list(passes),
        dataset_name=_string_or_none(record.get("dataset_name")),
        id_in_dataset=record.get("id_in_dataset"),
        total_passes=total_passes,
        error_count=sum(1 for item in passes if item.error_message),
        source=source,
        split=split,
        selected_passes=[item.pass_name for item in passes],
        fallback_warnings=list(fallback_warnings),
        run_config=dict(run_config),
        sample_duration_ms=sample_duration_ms,
    )


def assign_bucket(
    *,
    pass_success_count: int,
    total_passes: int,
    support_rate: float,
    max_node_count: int,
    max_labeled_token_count: int,
    exception_override: Mapping[str, Any] | None,
) -> tuple[str, str]:
    if exception_override is not None:
        if bool(exception_override.get("has_partial_graph")):
            return "threshold_b", f"sample_exception_partial_graph:{exception_override.get('type')}:{exception_override.get('message')}"
        return "hard_absent", f"sample_exception_no_graph:{exception_override.get('type')}:{exception_override.get('message')}"
    if total_passes > 0 and pass_success_count == total_passes and max_labeled_token_count > 0:
        return "stable_graph", "all_passes_succeeded_with_labels"
    if pass_success_count >= 2 or (support_rate >= 0.66 and max_node_count > 0):
        return "threshold_a", "multi_pass_support_or_high_support_rate"
    if pass_success_count == 1 or (max_node_count > 0 and max_labeled_token_count == 0):
        return "threshold_b", "single_pass_support_or_nodes_without_labels"
    if pass_success_count == 0 and max_node_count == 0:
        return "hard_absent", "no_graph_support_in_any_pass"
    if max_node_count > 0:
        return "threshold_b", "fallback_nodes_detected"
    return "hard_absent", "fallback_no_nodes_detected"


def extract_question_and_options(record: Mapping[str, Any]) -> tuple[str, list[str]]:
    question = _first_text(record, QUESTION_KEYS)
    options_value = record.get("options", record.get("choices"))
    return question, parse_options(options_value)


def render_question_with_options(question: str, option_texts: Sequence[str]) -> str:
    cleaned_question = str(question or "").strip()
    cleaned_options = [strip_mcq_prefixes(option) for option in option_texts if strip_mcq_prefixes(option)]
    if not cleaned_options:
        return cleaned_question
    return "\n".join([cleaned_question, "Options:", *cleaned_options]).strip()


def parse_options(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        options: list[str] = []
        for _, item in value.items():
            text = strip_mcq_prefixes(_flatten_to_text(item))
            if text:
                options.append(text)
        return options
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [text for item in value if (text := strip_mcq_prefixes(_flatten_to_text(item)))]
    text = str(value)
    lines = [line for line in strip_mcq_prefixes(text).splitlines() if line.strip()]
    if lines:
        return lines
    return [strip_mcq_prefixes(text)] if strip_mcq_prefixes(text) else []


def build_sample_id(record: Mapping[str, Any], row_index: int) -> str:
    dataset_name = _string_or_none(record.get("dataset_name"))
    id_in_dataset = record.get("id_in_dataset")
    if dataset_name is not None and id_in_dataset is not None:
        return f"{dataset_name}:{id_in_dataset}"
    for key in ("sample_id", "group_id", "id", "question_id", "uid"):
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return str(row_index)


def classify_question_type(question: str) -> str:
    normalized = str(question or "").lower()
    for question_type, patterns in QUESTION_TYPE_PATTERNS.items():
        if question_type in {"factoid", "other"}:
            continue
        if any(pattern in normalized for pattern in patterns):
            return str(question_type)
    return "factoid" if "factoid" in QUESTION_TYPE_PATTERNS else "other"


def default_relation_filter(question_type: str) -> set[str] | None:
    relation_filter = QUESTION_TYPE_RELATION_FILTERS.get(question_type)
    return set(relation_filter) if relation_filter else None


def link_entities(question_text: str, dependencies: RuntimeDependencies) -> list[Any]:
    with dependencies.linker_lock:
        linked = dependencies.linker.link(question_text)
    return merge_entities([], list(linked))


def resolve_entities(raw_entities: Sequence[Any], dependencies: RuntimeDependencies) -> list[Any]:
    resolved = dependencies.kg_extractor.resolve_seed_entities(list(raw_entities))
    return merge_entities([], list(resolved))


def augment_entities_broad(resolved_entities: Sequence[Any], *, dependencies: RuntimeDependencies) -> list[Any]:
    entities: list[Any] = []
    for entity in resolved_entities:
        if _entity_node_id(entity):
            entities.append(entity)
            continue
        matched_node_id = broad_match_entity_to_node(entity, dependencies=dependencies)
        entities.append(_copy_entity(entity, primekg_node_id=matched_node_id))
    return merge_entities([], entities)


def broad_match_entity_to_node(entity: Any, *, dependencies: RuntimeDependencies) -> str | None:
    surface = _normalize_name(_entity_surface(entity))
    if not surface:
        return None
    with dependencies.broad_match_cache_lock:
        if surface in dependencies.broad_match_cache:
            return dependencies.broad_match_cache[surface]

    node_name_index = getattr(dependencies.kg_extractor, "node_name_index", {}) or {}
    candidate_ids: set[str] = set(node_name_index.get(surface, set()))
    ensure_node_token_index(dependencies)
    token_index = dependencies.node_token_index or {}
    surface_tokens = {token for token in surface.split() if len(token) >= 3}
    for token in surface_tokens:
        candidate_ids.update(token_index.get(token, []))

    best_node_id: str | None = None
    best_score = 0.0
    for candidate_id in candidate_ids:
        candidate_name = node_name_for_id(dependencies.kg_extractor, candidate_id)
        candidate_norm = _normalize_name(candidate_name)
        if not candidate_norm:
            continue
        overlap = surface_tokens.intersection(set(candidate_norm.split()))
        score = 0.0
        if candidate_norm == surface:
            score += 100.0
        if candidate_norm.startswith(surface) or surface.startswith(candidate_norm):
            score += 20.0
        if surface in candidate_norm or candidate_norm in surface:
            score += 10.0
        score += len(overlap) / max(len(surface_tokens), 1) * 5.0
        if entity_matches_node_type(dependencies.kg_extractor, _entity_type(entity), candidate_id):
            score += 2.0
        score += min(float(getattr(dependencies.kg_extractor, "node_degrees", {}).get(candidate_id, 0)), 50.0) / 50.0
        if score > best_score:
            best_score = score
            best_node_id = candidate_id
    if best_score < 3.0:
        best_node_id = None
    with dependencies.broad_match_cache_lock:
        dependencies.broad_match_cache[surface] = best_node_id
    return best_node_id


def ensure_node_token_index(dependencies: RuntimeDependencies) -> None:
    if dependencies.node_token_index is not None:
        return
    with dependencies.token_index_lock:
        if dependencies.node_token_index is not None:
            return
        token_index: dict[str, list[str]] = {}
        node_name_index = getattr(dependencies.kg_extractor, "node_name_index", {}) or {}
        for normalized_name, node_ids in node_name_index.items():
            tokens = {token for token in str(normalized_name).split() if len(token) >= 3}
            if not tokens:
                continue
            for token in tokens:
                bucket = token_index.setdefault(token, [])
                if len(bucket) >= BROAD_TOKEN_CAP:
                    continue
                for node_id in node_ids:
                    if node_id in bucket:
                        continue
                    bucket.append(node_id)
                    if len(bucket) >= BROAD_TOKEN_CAP:
                        break
        dependencies.node_token_index = token_index


def estimate_candidate_path_count(
    *,
    kg_extractor: Any,
    resolved_entities: Sequence[Any],
    relation_filter: set[str] | None,
) -> int:
    if not hasattr(kg_extractor, "_iter_incident_edges"):
        return 0
    seed_node_ids = {_entity_node_id(entity) for entity in resolved_entities if _entity_node_id(entity)}
    if not seed_node_ids:
        return 0
    allowed_relations = {relation.lower() for relation in relation_filter} if relation_filter else None
    total_paths = 0
    for seed_node_id in seed_node_ids:
        hop1_edges = list(kg_extractor._iter_incident_edges(node_id=seed_node_id, allowed_relations=allowed_relations))
        total_paths += len(hop1_edges)
        if total_paths >= CANDIDATE_PATH_CAP:
            return CANDIDATE_PATH_CAP
        for edge in hop1_edges:
            intermediate_nodes = {getattr(edge, "head", None), getattr(edge, "tail", None)} - {seed_node_id, None}
            for intermediate in intermediate_nodes:
                hop2_edges = list(kg_extractor._iter_incident_edges(node_id=intermediate, allowed_relations=allowed_relations))
                total_paths += len([candidate for candidate in hop2_edges if getattr(candidate, "edge_id", None) != getattr(edge, "edge_id", None)])
                if total_paths >= CANDIDATE_PATH_CAP:
                    return CANDIDATE_PATH_CAP
    return total_paths


def extract_subgraph_edges(
    *,
    kg_extractor: Any,
    resolved_entities: Sequence[Any],
    top_k: int,
    relation_filter: set[str] | None,
) -> list[Any]:
    return list(kg_extractor.extract_2hop(seed_entities=list(resolved_entities), top_k=top_k, relation_filter=relation_filter))


def count_nodes(resolved_entities: Sequence[Any], subgraph_edges: Sequence[Any]) -> int:
    node_ids = {_entity_node_id(entity) for entity in resolved_entities if _entity_node_id(entity)}
    for edge in subgraph_edges:
        if getattr(edge, "head", None):
            node_ids.add(str(edge.head))
        if getattr(edge, "tail", None):
            node_ids.add(str(edge.tail))
    return len(node_ids)


def compute_labeled_token_count(
    *,
    row_index: int,
    record: Mapping[str, Any],
    question_text: str,
    question_type: str,
    question_entities: Sequence[Any],
    subgraph_edges: Sequence[Any],
    dependencies: RuntimeDependencies,
    normalize_text: bool,
) -> int:
    del normalize_text
    if dependencies.embedder is None or path_to_token_sequence is None or map_medreason_edges is None:
        return 0
    if not subgraph_edges:
        return 0
    evidence = build_evidence_bundle(
        question_text=question_text,
        question_type=question_type,
        question_entities=question_entities,
        subgraph_edges=subgraph_edges,
        metadata={
            "question_type": question_type,
            "entity_count": len(question_entities),
            "edge_count": len(subgraph_edges),
            "pubmed_count": 0,
            "entity_linker_backend": getattr(dependencies.linker, "backend_used", "unknown"),
            "primekg_backend": getattr(dependencies.kg_extractor, "backend_used", "unknown"),
            "backend_used": {
                "entity_linker": getattr(dependencies.linker, "backend_used", "unknown"),
                "primekg": getattr(dependencies.kg_extractor, "backend_used", "unknown"),
                "pubmed": "disabled",
            },
        },
    )
    if evidence is None:
        return 0
    example = build_example(record=record, row_index=row_index)
    try:
        gold_edge_ids, _ = map_medreason_edges(example=example, evidence=evidence, backend="heuristic", selector=None)
    except Exception:
        return 0
    if not gold_edge_ids:
        return 0
    with dependencies.embedder_lock:
        encoded = dependencies.embedder(evidence)
    _, diagnostics = path_to_token_sequence(gold_edge_ids, encoded, evidence)
    return int(diagnostics.get("labeled_token_count", 0))


def build_evidence_bundle(
    *,
    question_text: str,
    question_type: str,
    question_entities: Sequence[Any],
    subgraph_edges: Sequence[Any],
    metadata: Mapping[str, Any],
) -> Any:
    if EvidenceBundle is None:
        return None
    payload = {
        "question_text": question_text,
        "question_type": question_type if question_type in QUESTION_TYPE_RELATION_FILTERS else "other",
        "question_entities": [entity_to_payload(entity) for entity in question_entities],
        "subgraph_edges": [edge_to_payload(edge) for edge in subgraph_edges],
        "pubmed_passages": [],
        "metadata": dict(metadata),
    }
    return EvidenceBundle.model_validate(payload)


def build_example(*, record: Mapping[str, Any], row_index: int) -> LocalExample | Any:
    if normalize_medreason_record is not None:
        try:
            return normalize_medreason_record(record, index=row_index)
        except Exception:
            pass
    question, options = extract_question_and_options(record)
    rendered_question = render_question_with_options(question, options)
    answer = _first_text(record, ANSWER_KEYS)
    reasoning = _first_value(record, REASONING_KEYS)
    if reasoning is None:
        reasoning = answer or rendered_question
    return LocalExample(
        question=rendered_question,
        answer=answer,
        reasoning=reasoning,
        direct_edge_ids=parse_reasoning_edge_ids(reasoning),
        group_id=build_sample_id(record, row_index),
        raw=dict(record),
    )


def parse_reasoning_edge_ids(reasoning: Any) -> list[str]:
    edge_ids: list[str] = []

    def add_edge_id(value: Any) -> None:
        if value is None:
            return
        text = str(value).strip().removeprefix("edge:").removeprefix("edge_id:")
        if text and text not in edge_ids:
            edge_ids.append(text)

    if isinstance(reasoning, Mapping):
        for key in ("gold_edge_ids", "edge_ids", "premise_edge_ids"):
            value = reasoning.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for item in value:
                    add_edge_id(item)
            else:
                add_edge_id(value)
        for value in reasoning.values():
            for edge_id in parse_reasoning_edge_ids(value):
                add_edge_id(edge_id)
        return edge_ids
    if isinstance(reasoning, Sequence) and not isinstance(reasoning, (str, bytes)):
        for item in reasoning:
            for edge_id in parse_reasoning_edge_ids(item):
                add_edge_id(edge_id)
        return edge_ids
    text = str(reasoning or "")
    for match in EDGE_ID_PATTERN.finditer(text):
        add_edge_id(match.group(1) or match.group(2))
    return edge_ids


def write_jsonl_append(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, payload)


def persist_batch(
    *,
    batch: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    csv_path: Path,
    bucket_paths: Mapping[str, Path],
    progress_path: Path,
    summary_path: Path,
    state: PersistedState,
    logger: logging.Logger,
) -> None:
    write_jsonl_append(manifest_path, batch)
    write_bucket_batches(bucket_paths=bucket_paths, batch=batch)
    append_manifest_csv(csv_path=csv_path, rows=batch)
    updated_at = utc_now_iso()
    summary_payload = build_summary_payload(state=state, updated_at=updated_at)
    progress_payload = build_progress_payload(state=state, updated_at=updated_at)
    atomic_write_json(summary_path, summary_payload)
    write_progress(progress_path, progress_payload)
    logger.info("Persisted %s triage rows. total_done=%s", len(batch), state.total_done)


def write_bucket_batches(*, bucket_paths: Mapping[str, Path], batch: Sequence[Mapping[str, Any]]) -> None:
    for bucket in VALID_BUCKETS:
        bucket_paths[bucket].parent.mkdir(parents=True, exist_ok=True)
        bucket_paths[bucket].touch(exist_ok=True)
        rows = [row for row in batch if row.get("bucket") == bucket]
        if rows:
            write_jsonl_append(bucket_paths[bucket], rows)


def append_manifest_csv(*, csv_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_HEADERS))
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(flatten_manifest_row(row))
        handle.flush()
        os.fsync(handle.fileno())


def flatten_manifest_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in CSV_HEADERS}


def update_persisted_state(state: PersistedState, rows: Sequence[Mapping[str, Any]]) -> PersistedState:
    for row in rows:
        bucket = str(row.get("bucket"))
        state.total_done += 1
        state.last_row_index = int(row.get("row_index", state.last_row_index))
        state.duration_sum_ms += float(row.get("sample_duration_ms") or 0.0)
        if bucket in state.bucket_counts:
            state.bucket_counts[bucket] += 1
        else:
            state.bucket_counts[bucket] = 1
        if int(row.get("error_count") or 0) > 0:
            state.total_errors += 1
        state.skip_row_indexes.add(int(row.get("row_index")))
        state.skip_sample_ids.add(str(row.get("sample_id")))
        if not state.run_config and isinstance(row.get("run_config"), Mapping):
            state.run_config = dict(row["run_config"])
    return state


def build_progress_payload(*, state: PersistedState, updated_at: str) -> dict[str, Any]:
    return {
        "total_seen": state.total_seen,
        "total_done": state.total_done,
        "total_errors": state.total_errors,
        "last_row_index": state.last_row_index,
        "started_at": state.started_at,
        "updated_at": updated_at,
        "bucket_counts": dict(state.bucket_counts),
        "elapsed_sec": elapsed_seconds(state.started_at, updated_at),
    }


def build_summary_payload(*, state: PersistedState, updated_at: str) -> dict[str, Any]:
    return {
        "started_at": state.started_at,
        "updated_at": updated_at,
        "elapsed_sec": elapsed_seconds(state.started_at, updated_at),
        "total_target": state.total_target,
        "total_done": state.total_done,
        "total_errors": state.total_errors,
        "bucket_counts": dict(state.bucket_counts),
        "avg_sample_duration_ms": round(state.avg_ms, 3),
        "run_config": dict(state.run_config),
    }


def validate_outputs(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "triage_manifest.jsonl"
    csv_path = output_dir / "triage_manifest.csv"
    summary_path = output_dir / "triage_summary.json"
    progress_path = output_dir / "progress.json"
    bucket_paths = {bucket: output_dir / f"bucket_{bucket}.jsonl" for bucket in VALID_BUCKETS}

    messages: list[str] = []
    if not manifest_path.exists():
        return {"valid": False, "messages": [f"Missing manifest: {manifest_path}"]}

    manifest_count = 0
    seen_row_indexes: set[int] = set()
    manifest_bucket_counts = Counter()
    progress_bucket_counts = Counter()
    summary_bucket_counts = Counter()
    csv_count = 0

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            bucket = row.get("bucket")
            row_index = row.get("row_index")
            manifest_count += 1
            if bucket not in VALID_BUCKETS:
                messages.append(f"Invalid bucket at manifest line {line_number}: {bucket!r}")
            else:
                manifest_bucket_counts[str(bucket)] += 1
            if row_index in seen_row_indexes:
                messages.append(f"Duplicate row_index in manifest: {row_index}")
            else:
                seen_row_indexes.add(int(row_index))

    bucket_file_counts = Counter()
    for bucket, bucket_path in bucket_paths.items():
        if not bucket_path.exists():
            messages.append(f"Missing bucket file: {bucket_path.name}")
            continue
        with bucket_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                bucket_file_counts[bucket] += 1
                if row.get("bucket") != bucket:
                    messages.append(f"Bucket mismatch in {bucket_path.name} line {line_number}: {row.get('bucket')!r}")

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            csv_count = sum(1 for _ in reader)
    else:
        messages.append(f"Missing CSV manifest: {csv_path.name}")

    if summary_path.exists():
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_bucket_counts.update(summary_payload.get("bucket_counts", {}))
    else:
        messages.append(f"Missing summary file: {summary_path.name}")

    if progress_path.exists():
        progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
        progress_bucket_counts.update(progress_payload.get("bucket_counts", {}))
    else:
        messages.append(f"Missing progress file: {progress_path.name}")

    expected_bucket_counts = {bucket: manifest_bucket_counts.get(bucket, 0) for bucket in VALID_BUCKETS}
    actual_bucket_counts = {bucket: bucket_file_counts.get(bucket, 0) for bucket in VALID_BUCKETS}
    summary_counts = {bucket: summary_bucket_counts.get(bucket, 0) for bucket in VALID_BUCKETS}
    progress_counts = {bucket: progress_bucket_counts.get(bucket, 0) for bucket in VALID_BUCKETS}

    if manifest_count != sum(bucket_file_counts.values()):
        messages.append(
            f"Manifest count {manifest_count} does not equal total bucket records {sum(bucket_file_counts.values())}."
        )
    if csv_count != manifest_count:
        messages.append(f"Manifest CSV row count {csv_count} does not equal manifest count {manifest_count}.")
    if expected_bucket_counts != actual_bucket_counts:
        messages.append("Manifest bucket counts do not match bucket file counts.")
    if summary_bucket_counts and summary_counts != expected_bucket_counts:
        messages.append("Summary bucket counts do not match manifest bucket counts.")
    if progress_bucket_counts and progress_counts != expected_bucket_counts:
        messages.append("Progress bucket counts do not match manifest bucket counts.")

    return {
        "valid": not messages,
        "messages": messages,
        "manifest_count": manifest_count,
        "bucket_counts": expected_bucket_counts,
        "bucket_file_counts": actual_bucket_counts,
        "csv_count": csv_count,
        "duplicate_row_index_count": manifest_count - len(seen_row_indexes),
    }


def reconcile_outputs_from_manifest(
    *,
    output_dir: Path,
    started_at: str,
    target_total: int,
    logger: logging.Logger,
) -> PersistedState:
    manifest_path = output_dir / "triage_manifest.jsonl"
    bucket_paths = {bucket: output_dir / f"bucket_{bucket}.jsonl" for bucket in VALID_BUCKETS}
    csv_path = output_dir / "triage_manifest.csv"
    summary_path = output_dir / "triage_summary.json"
    progress_path = output_dir / "progress.json"
    state = PersistedState(started_at=started_at, updated_at=utc_now_iso(), total_target=target_total)

    temp_paths: list[Path] = []
    csv_temp = temporary_path(csv_path)
    temp_paths.append(csv_temp)
    bucket_temps = {bucket: temporary_path(path) for bucket, path in bucket_paths.items()}
    temp_paths.extend(bucket_temps.values())

    with (
        manifest_path.open("r", encoding="utf-8") as manifest_handle,
        csv_temp.open("w", encoding="utf-8", newline="") as csv_handle,
        bucket_temps["stable_graph"].open("w", encoding="utf-8") as stable_handle,
        bucket_temps["threshold_a"].open("w", encoding="utf-8") as a_handle,
        bucket_temps["threshold_b"].open("w", encoding="utf-8") as b_handle,
        bucket_temps["hard_absent"].open("w", encoding="utf-8") as absent_handle,
    ):
        bucket_handles = {
            "stable_graph": stable_handle,
            "threshold_a": a_handle,
            "threshold_b": b_handle,
            "hard_absent": absent_handle,
        }
        writer = csv.DictWriter(csv_handle, fieldnames=list(CSV_HEADERS))
        writer.writeheader()
        for line_number, line in enumerate(manifest_handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_index = int(row["row_index"])
            sample_id = str(row["sample_id"])
            if row_index in state.skip_row_indexes:
                raise ValueError(f"Duplicate row_index found while reconciling manifest: {row_index} (line {line_number})")
            state.skip_row_indexes.add(row_index)
            state.skip_sample_ids.add(sample_id)
            update_persisted_state(state, [row])
            writer.writerow(flatten_manifest_row(row))
            bucket = str(row["bucket"])
            if bucket not in bucket_handles:
                raise ValueError(f"Invalid bucket in manifest line {line_number}: {bucket!r}")
            bucket_handles[bucket].write(json.dumps(json_safe(row), ensure_ascii=True, sort_keys=True) + "\n")
            if not state.run_config and isinstance(row.get("run_config"), Mapping):
                state.run_config = dict(row["run_config"])
        csv_handle.flush()
        os.fsync(csv_handle.fileno())
        for handle in bucket_handles.values():
            handle.flush()
            os.fsync(handle.fileno())

    os.replace(csv_temp, csv_path)
    for bucket, temp_path in bucket_temps.items():
        os.replace(temp_path, bucket_paths[bucket])
    updated_at = utc_now_iso()
    atomic_write_json(summary_path, build_summary_payload(state=state, updated_at=updated_at))
    write_progress(progress_path, build_progress_payload(state=state, updated_at=updated_at))
    logger.info("Reconciled derived outputs from manifest. total_done=%s", state.total_done)
    return state


def read_existing_started_at(progress_path: Path) -> str | None:
    if not progress_path.exists():
        return None
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    started_at = payload.get("started_at")
    return str(started_at) if started_at else None


def build_final_summary(*, state: PersistedState, output_dir: Path, validation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "output_dir": str(output_dir),
        "total_done": state.total_done,
        "total_errors": state.total_errors,
        "bucket_counts": dict(state.bucket_counts),
        "avg_sample_duration_ms": round(state.avg_ms, 3),
        "validation_valid": bool(validation.get("valid")),
        "validation_messages": list(validation.get("messages", [])),
    }


def print_progress(state: PersistedState) -> None:
    target = max(state.total_target, state.total_done)
    print(
        f"[{state.total_done}/{target}] "
        f"stable={state.bucket_counts.get('stable_graph', 0)} "
        f"a={state.bucket_counts.get('threshold_a', 0)} "
        f"b={state.bucket_counts.get('threshold_b', 0)} "
        f"absent={state.bucket_counts.get('hard_absent', 0)} "
        f"errors={state.total_errors} "
        f"avg_ms={state.avg_ms:.2f}"
    )


def temporary_path(path: Path) -> Path:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    return Path(temp_name)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temp_path = temporary_path(path)
    try:
        temp_path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def elapsed_seconds(started_at: str, updated_at: str) -> float:
    try:
        started = datetime.fromisoformat(started_at)
        updated = datetime.fromisoformat(updated_at)
    except ValueError:
        return 0.0
    return max((updated - started).total_seconds(), 0.0)


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def merge_entities(base_entities: Sequence[Any], extra_entities: Sequence[Any]) -> list[Any]:
    merged: list[Any] = []
    seen: set[tuple[str, str | None, str]] = set()
    for entity in [*base_entities, *extra_entities]:
        key = (_normalize_name(_entity_surface(entity)), _entity_cui(entity), _entity_type(entity))
        if key in seen:
            continue
        seen.add(key)
        merged.append(entity)
    return merged


def entity_to_payload(entity: Any) -> dict[str, Any]:
    return {
        "surface": _entity_surface(entity),
        "cui": _entity_cui(entity),
        "primekg_node_id": _entity_node_id(entity),
        "entity_type": _entity_type(entity),
    }


def edge_to_payload(edge: Any) -> dict[str, Any]:
    payload = {
        "edge_id": getattr(edge, "edge_id", ""),
        "head": getattr(edge, "head", ""),
        "tail": getattr(edge, "tail", ""),
        "relation": getattr(edge, "relation", "disease_disease"),
        "display_relation": getattr(edge, "display_relation", getattr(edge, "relation", "")),
        "source_reliability": float(getattr(edge, "source_reliability", 0.0) or 0.0),
        "amg_confidence": float(getattr(edge, "amg_confidence", 1.0) or 1.0),
        "supporting_pmids": list(getattr(edge, "supporting_pmids", []) or []),
    }
    if not payload["display_relation"]:
        payload["display_relation"] = str(payload["relation"]).replace("_", " ")
    return payload


def _copy_entity(entity: Any, *, primekg_node_id: str | None) -> Any:
    if hasattr(entity, "model_copy"):
        return entity.model_copy(update={"primekg_node_id": primekg_node_id})
    return SimpleEntity(
        surface=_entity_surface(entity),
        cui=_entity_cui(entity),
        primekg_node_id=primekg_node_id,
        entity_type=_entity_type(entity),
    )


def _entity_surface(entity: Any) -> str:
    return str(getattr(entity, "surface", "") or "")


def _entity_cui(entity: Any) -> str | None:
    value = getattr(entity, "cui", None)
    return str(value) if value is not None else None


def _entity_type(entity: Any) -> str:
    value = getattr(entity, "entity_type", "other")
    return str(value or "other")


def _entity_node_id(entity: Any) -> str | None:
    value = getattr(entity, "primekg_node_id", None)
    return str(value) if value is not None else None


def node_name_for_id(kg_extractor: Any, node_id: str) -> str:
    graph = getattr(kg_extractor, "graph", None)
    if graph is None or not hasattr(graph, "nodes"):
        return node_id
    try:
        node_attrs = graph.nodes[node_id]
    except Exception:
        return node_id
    return str(node_attrs.get("name") or node_id)


def entity_matches_node_type(kg_extractor: Any, entity_type: str, candidate_id: str) -> bool:
    graph = getattr(kg_extractor, "graph", None)
    if graph is None or not hasattr(graph, "nodes"):
        return entity_type == "other"
    try:
        node_type = str(graph.nodes[candidate_id].get("node_type", "")).lower()
    except Exception:
        return entity_type == "other"
    if hasattr(kg_extractor, "_entity_matches_node_type"):
        try:
            return bool(kg_extractor._entity_matches_node_type(entity_type, node_type))
        except Exception:
            return entity_type == node_type or entity_type == "other"
    return entity_type == node_type or entity_type == "other"


def _infer_entity_type(surface: str) -> str:
    lowered = _normalize_name(surface)
    if lowered.endswith(("mab", "nib", "statin", "pril", "sartan", "olol")):
        return "drug"
    if lowered in {"pain", "rash", "fever", "cough", "nausea"}:
        return "symptom"
    if lowered in {"heart", "lung", "liver", "kidney", "brain"}:
        return "anatomy"
    if lowered.isupper() and 2 <= len(lowered) <= 8:
        return "protein"
    return "other"


def _iter_candidate_phrases(text: str) -> Iterator[str]:
    tokens = re.findall(r"[A-Za-z0-9-]+", text.lower())
    stopwords = {"a", "an", "and", "are", "be", "for", "from", "how", "if", "in", "is", "of", "on", "or", "the", "to", "what", "when", "with"}
    for ngram_size in range(3, 0, -1):
        for start_index in range(0, len(tokens) - ngram_size + 1):
            ngram_tokens = tokens[start_index : start_index + ngram_size]
            if any(token in stopwords for token in ngram_tokens):
                continue
            phrase = " ".join(ngram_tokens)
            if len(phrase) >= 3:
                yield phrase


def _normalize_name(text: str) -> str:
    lowered = str(text or "").strip().lower().replace("_", " ").replace("-", " ")
    lowered = PUNCT_CLEAN_PATTERN.sub(" ", lowered)
    return WHITESPACE_PATTERN.sub(" ", lowered).strip()


def _first_text(record: Mapping[str, Any], keys: Sequence[str]) -> str:
    value = _first_value(record, keys)
    return _flatten_to_text(value).strip()


def _first_value(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = _flatten_to_text(value).strip()
        if text:
            return value
    return None


def _flatten_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        parts = [_flatten_to_text(item) for item in value.values()]
        return " ".join(part for part in parts if part)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = [_flatten_to_text(item) for item in value]
        return " ".join(part for part in parts if part)
    return str(value).strip()


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


if __name__ == "__main__":
    raise SystemExit(main())
