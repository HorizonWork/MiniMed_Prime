from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, Field

from src.layers.layer1_retrieval import AgenticRetriever
from src.retrieval import (
    DEFAULT_GEMINI_GROUNDER,
    DEFAULT_OPENAI_GROUNDER,
    DeterministicSeedGrounder,
    GROUNDING_MODES,
    LightweightSeedEntityExtractor,
    LLMAssistedSeedGrounder,
    PrimeKGNodeCatalog,
    SeedEntityExtractor,
    SeedGrounder,
    infer_question_type,
    normalize_basic_text,
    serialize_grounding_decisions,
    summarize_grounding_candidates,
)
from src.layers.layer4_judges import JudgeBase
from src.schemas import EvidenceBundle, KGEdge, PubMedPassage, QuestionEntity
from src.training.trm_dataset_builder import parse_reasoning_chain
from src.utils.kaggle_env import KaggleEnv
from src.utils.structured_logger import DEFAULT_LOG_DIR, StructuredLogger


DEFAULT_MEDREASON_SOURCE = "data/medreason"
HF_MEDREASON_SOURCE = "UCSC-VLAA/MedReason"
DEFAULT_OPENAI_EDGE_SELECTOR = "gpt-4o-mini"
DEFAULT_GEMINI_EDGE_SELECTOR = "gemini-2.5-flash"
DEFAULT_HEURISTIC_SELECTOR_NAME = "heuristic"
DEFAULT_GROUNDING_MODE = "baseline_current"
DEFAULT_LOCAL_TRAIN_FRACTION = 0.90
DEFAULT_LOCAL_VALIDATION_FRACTION = 0.05
DEFAULT_MAX_LLM_CANDIDATE_EDGES = 80
DEFAULT_HEURISTIC_EDGE_THRESHOLD = 3.0
SUPPORTED_LOCAL_EXTENSIONS = (".jsonl", ".json", ".csv", ".parquet")
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


class EdgeSelection(BaseModel):
    edge_ids: list[str] = Field(default_factory=list)
    rationale: str = ""


@dataclass(slots=True)
class RawMedReasonExample:
    question: str
    answer: str
    reasoning: Any
    direct_edge_ids: list[str]
    group_id: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EdgeMappingDiagnostics:
    direct_edge_ids: list[str]
    direct_edge_ids_in_evidence: list[str]
    missing_direct_edge_ids: list[str]
    heuristic_edge_ids: list[str]
    selected_edge_ids: list[str]
    backend_used: str
    llm_rationale: str = ""


@dataclass(slots=True)
class MedReasonSeedResult:
    output_path: Path
    skipped_path: Path | None
    records_seen: int
    records_saved: int
    records_skipped: int
    backend_used: str
    latency_ms: float
    skip_stage_counts: dict[str, int] = field(default_factory=dict)
    skip_reason_counts: dict[str, int] = field(default_factory=dict)
    skipped_avg_edge_count: float = 0.0
    skipped_avg_entity_count: float = 0.0

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_path"] = str(self.output_path)
        payload["skipped_path"] = str(self.skipped_path) if self.skipped_path is not None else None
        return payload


class MedReasonEdgeSelector(JudgeBase):
    def __init__(self, model_name: str = "heuristic", device: str = "cpu") -> None:
        super().__init__(
            model_name=model_name,
            system_prompt=(
                "You map MedReason chain-of-thought evidence to PrimeKG edge IDs. "
                "Select only candidate edge IDs that are explicitly supported by the reasoning text. "
                "Return strict JSON only."
            ),
            device=device,
        )

    def evaluate(self, **kwargs: Any) -> EdgeSelection:
        prompt = str(kwargs["prompt"])
        fallback_edge_ids = list(kwargs.get("fallback_edge_ids", []))
        return self._call_llm(
            prompt=prompt,
            schema=EdgeSelection,
            fallback_payload={"edge_ids": fallback_edge_ids, "rationale": "heuristic fallback"},
        )


def load_medreason_records(
    source: str | Path = DEFAULT_MEDREASON_SOURCE,
    *,
    split: str | None = "train",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    source_path = _resolve_source_path(source)
    if source_path is not None and source_path.exists():
        records = _load_local_records(source_path, split=split)
    else:
        records = _load_hf_records(str(source), split=split)
    if limit is not None:
        records = records[:limit]
    return records


def normalize_medreason_record(record: Mapping[str, Any], *, index: int = 0) -> RawMedReasonExample:
    question = _first_text(record, QUESTION_KEYS)
    if not question:
        raise ValueError("MedReason record did not include a recognizable question field.")
    options_text = _options_text(record.get("options", record.get("choices")))
    if options_text and options_text.lower() not in question.lower():
        question = f"{question}\nOptions: {options_text}"
    answer = _first_text(record, ANSWER_KEYS)
    reasoning = _first_value(record, REASONING_KEYS)
    if reasoning is None:
        reasoning = answer or question
    direct_edge_ids = parse_reasoning_chain(record)
    group_id = str(
        record.get("id")
        or record.get("question_id")
        or record.get("uid")
        or record.get("sample_id")
        or index
    )
    return RawMedReasonExample(
        question=question,
        answer=answer,
        reasoning=reasoning,
        direct_edge_ids=direct_edge_ids,
        group_id=group_id,
        raw=dict(record),
    )


def prepare_medreason_seed_jsonl(
    *,
    source: str | Path = DEFAULT_MEDREASON_SOURCE,
    output_jsonl: Path,
    retriever: AgenticRetriever,
    split: str | None = "train",
    limit: int | None = None,
    max_saved: int | None = None,
    edge_mapper_backend: str = "auto",
    llm_model_name: str = DEFAULT_HEURISTIC_SELECTOR_NAME,
    llm_device: str = "cpu",
    grounding_mode: str = DEFAULT_GROUNDING_MODE,
    allow_empty_gold: bool = False,
) -> MedReasonSeedResult:
    started_at = time.perf_counter()
    logger = StructuredLogger("medreason_adapter", DEFAULT_LOG_DIR)
    output_path = KaggleEnv.ensure_writeable(output_jsonl if output_jsonl.is_absolute() else KaggleEnv.path(output_jsonl))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_path = output_path.with_suffix(".skipped.jsonl")
    records = load_medreason_records(source, split=split, limit=limit)
    effective_backend, effective_llm_model_name = resolve_edge_mapper_config(edge_mapper_backend, llm_model_name)
    normalized_grounding_mode, effective_grounding_llm_model = resolve_grounding_mode_config(grounding_mode, llm_model_name)
    selector = (
        MedReasonEdgeSelector(model_name=effective_llm_model_name, device=llm_device)
        if effective_backend == "llm"
        else None
    )
    node_catalog = (
        PrimeKGNodeCatalog.from_primekg_path(retriever.config.primekg_path)
        if normalized_grounding_mode != "baseline_current"
        else None
    )
    seed_grounder = (
        _build_seed_grounder(
            grounding_mode=normalized_grounding_mode,
            catalog=node_catalog,
            llm_model_name=effective_grounding_llm_model,
            llm_device=llm_device,
        )
        if node_catalog is not None
        else None
    )
    seed_entity_extractor = LightweightSeedEntityExtractor(node_catalog) if node_catalog is not None else None

    records_saved = 0
    skipped_rows: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for record_index, record in enumerate(records):
            if max_saved is not None and records_saved >= max_saved:
                break
            example: RawMedReasonExample | None = None
            evidence: EvidenceBundle | None = None
            diagnostics: EdgeMappingDiagnostics | None = None
            skip_stage = "export"
            try:
                skip_stage = "export"
                example = normalize_medreason_record(record, index=record_index)
                question_type_hint = _question_type_hint_from_record(record, question=example.question)
                skip_stage = "retrieval"
                evidence = build_seed_evidence_bundle(
                    question=example.question,
                    retriever=retriever,
                    question_type_hint=question_type_hint,
                    grounding_mode=normalized_grounding_mode,
                    llm_model_name=effective_grounding_llm_model,
                    llm_device=llm_device,
                    node_catalog=node_catalog,
                    seed_grounder=seed_grounder,
                    seed_entity_extractor=seed_entity_extractor,
                )
                skip_stage = "edge_mapping"
                edge_ids, diagnostics = map_medreason_edges(
                    example=example,
                    evidence=evidence,
                    backend=effective_backend,
                    selector=selector,
                )
                if not edge_ids and not allow_empty_gold:
                    skipped_row = _build_skipped_seed_row(
                        index=record_index,
                        raw_record=record,
                        example=example,
                        evidence=evidence,
                        diagnostics=diagnostics,
                        skip_stage=_infer_no_gold_skip_stage(evidence),
                        skip_reason="no_gold_edges_mapped",
                        requested_backend=edge_mapper_backend,
                        effective_backend=effective_backend,
                    )
                    skipped_rows.append(skipped_row)
                    logger.log_event(
                        "medreason_record_skipped",
                        {
                            **_loggable_skip_payload(skipped_row),
                            "latency_ms": 0.0,
                        },
                    )
                    continue
                payload = {
                    "evidence": evidence.model_dump(mode="json"),
                    "gold_edge_ids": edge_ids,
                    "answer": example.answer,
                    "reasoning": example.reasoning,
                    "group_id": example.group_id,
                    "metadata": {
                        "source": str(source),
                        "split": split,
                        "record_index": record_index,
                        "grounding_mode": normalized_grounding_mode,
                        "grounding_backend": evidence.metadata.get("grounding_backend"),
                        "entity_grounding": evidence.metadata.get("entity_grounding"),
                        "unresolved_entities": evidence.metadata.get("unresolved_entities"),
                        "seed_node_ids": evidence.metadata.get("seed_node_ids"),
                        "candidate_stats": evidence.metadata.get("candidate_stats"),
                        "llm_grounding_used": evidence.metadata.get("llm_grounding_used"),
                        "llm_grounding_fallback_reason": evidence.metadata.get("llm_grounding_fallback_reason"),
                        "edge_mapping": asdict(diagnostics),
                    },
                }
                handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
                records_saved += 1
                logger.log_event(
                    "medreason_record_prepared",
                    {
                        "index": record_index,
                        "group_id": example.group_id,
                        "gold_edge_count": len(edge_ids),
                        "evidence_edge_count": len(evidence.subgraph_edges),
                        "pubmed_count": len(evidence.pubmed_passages),
                        "backend_used": diagnostics.backend_used,
                        "latency_ms": 0.0,
                    },
                )
            except Exception as exc:
                skipped_row = _build_skipped_seed_row(
                    index=record_index,
                    raw_record=record,
                    example=example,
                    evidence=evidence,
                    diagnostics=diagnostics,
                    skip_stage=skip_stage,
                    skip_reason=type(exc).__name__,
                    requested_backend=edge_mapper_backend,
                    effective_backend=effective_backend,
                    message=str(exc),
                )
                skipped_rows.append(skipped_row)
                logger.log_exception(
                    "medreason_record_failed",
                    exc,
                    {
                        **_loggable_skip_payload(skipped_row),
                        "backend_used": skipped_row["backend_used"],
                        "latency_ms": 0.0,
                    },
                )

    if skipped_rows:
        skipped_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in skipped_rows) + "\n",
            encoding="utf-8",
        )
    else:
        skipped_path = None

    skip_summary = _summarize_skipped_rows(skipped_rows)
    result = MedReasonSeedResult(
        output_path=output_path,
        skipped_path=skipped_path,
        records_seen=len(records),
        records_saved=records_saved,
        records_skipped=len(skipped_rows),
        backend_used=effective_backend,
        latency_ms=(time.perf_counter() - started_at) * 1000.0,
        skip_stage_counts=skip_summary["skip_stage_counts"],
        skip_reason_counts=skip_summary["skip_reason_counts"],
        skipped_avg_edge_count=skip_summary["skipped_avg_edge_count"],
        skipped_avg_entity_count=skip_summary["skipped_avg_entity_count"],
    )
    logger.log_event(
        "medreason_seed_saved",
        {
            **result.to_json_dict(),
            "backend_used": effective_backend,
            "requested_backend": edge_mapper_backend,
            "llm_model_name": effective_llm_model_name if effective_backend == "llm" else None,
            "grounding_mode": normalized_grounding_mode,
            "grounding_llm_model_name": (
                effective_grounding_llm_model if normalized_grounding_mode == "llm_assisted_v1" else None
            ),
            "latency_ms": result.latency_ms,
        },
    )
    return result


def resolve_edge_mapper_config(edge_mapper_backend: str, llm_model_name: str) -> tuple[str, str]:
    requested_backend = edge_mapper_backend.lower().strip()
    requested_model = _normalize_llm_model_alias((llm_model_name or DEFAULT_HEURISTIC_SELECTOR_NAME).strip())
    if requested_backend not in {"auto", "direct", "heuristic", "llm"}:
        raise ValueError("edge mapper backend must be one of: auto, direct, heuristic, llm.")

    if requested_backend == "auto":
        if _is_real_llm_model_name(requested_model):
            return "llm", requested_model
        if os.getenv("OPENAI_API_KEY"):
            return "llm", DEFAULT_OPENAI_EDGE_SELECTOR
        if os.getenv("GEMINI_API_KEY"):
            return "llm", DEFAULT_GEMINI_EDGE_SELECTOR
        return "heuristic", DEFAULT_HEURISTIC_SELECTOR_NAME

    if requested_backend == "llm" and not _is_real_llm_model_name(requested_model):
        if os.getenv("OPENAI_API_KEY"):
            return "llm", DEFAULT_OPENAI_EDGE_SELECTOR
        if os.getenv("GEMINI_API_KEY"):
            return "llm", DEFAULT_GEMINI_EDGE_SELECTOR
    return requested_backend, requested_model


def resolve_grounding_mode_config(grounding_mode: str, llm_model_name: str) -> tuple[str, str]:
    normalized_mode = grounding_mode.strip().lower()
    if normalized_mode not in GROUNDING_MODES:
        raise ValueError(f"grounding mode must be one of: {', '.join(GROUNDING_MODES)}.")
    requested_model = _normalize_llm_model_alias((llm_model_name or DEFAULT_HEURISTIC_SELECTOR_NAME).strip())
    if normalized_mode != "llm_assisted_v1":
        return normalized_mode, requested_model
    if _is_real_llm_model_name(requested_model):
        return normalized_mode, requested_model
    if os.getenv("OPENAI_API_KEY"):
        return normalized_mode, DEFAULT_OPENAI_GROUNDER
    if os.getenv("GEMINI_API_KEY"):
        return normalized_mode, DEFAULT_GEMINI_GROUNDER
    return normalized_mode, DEFAULT_HEURISTIC_SELECTOR_NAME


def build_seed_evidence_bundle(
    *,
    question: str,
    retriever: AgenticRetriever,
    question_type_hint: str | None = None,
    grounding_mode: str = DEFAULT_GROUNDING_MODE,
    llm_model_name: str = DEFAULT_HEURISTIC_SELECTOR_NAME,
    llm_device: str = "cpu",
    node_catalog: PrimeKGNodeCatalog | None = None,
    seed_grounder: SeedGrounder | None = None,
    seed_entity_extractor: SeedEntityExtractor | None = None,
) -> EvidenceBundle:
    normalized_mode, effective_model_name = resolve_grounding_mode_config(grounding_mode, llm_model_name)
    if normalized_mode == "baseline_current":
        evidence = retriever.retrieve(question, question_type_hint=question_type_hint)
        seed_node_ids = [entity.primekg_node_id for entity in evidence.question_entities if entity.primekg_node_id]
        unresolved_entities = [entity.surface for entity in evidence.question_entities if not entity.primekg_node_id]
        evidence.metadata.setdefault("grounding_mode", normalized_mode)
        evidence.metadata.setdefault("grounding_backend", "baseline_current")
        evidence.metadata.setdefault("entity_grounding", [])
        evidence.metadata.setdefault("unresolved_entities", unresolved_entities)
        evidence.metadata.setdefault("seed_node_ids", seed_node_ids)
        evidence.metadata.setdefault(
            "candidate_stats",
            {
                "entity_count": len(evidence.question_entities),
                "linked_count": len(seed_node_ids),
                "unresolved_count": len(unresolved_entities),
                "avg_candidates": 0.0,
                "max_candidates": 0,
                "fuzzy_link_count": 0,
            },
        )
        evidence.metadata.setdefault("llm_grounding_used", False)
        evidence.metadata.setdefault("llm_grounding_fallback_reason", None)
        return evidence

    if not hasattr(retriever, "kg_extractor"):
        raise ValueError("Data-only grounding requires a retriever with a kg_extractor component.")

    question_type = (
        infer_question_type(record={"question_type": question_type_hint}, question=question, fallback_label="factoid")
        if question_type_hint
        else infer_question_type(question=question, fallback_label="factoid")
    )
    catalog = node_catalog or PrimeKGNodeCatalog.from_primekg_path(retriever.config.primekg_path)
    entity_extractor = seed_entity_extractor or LightweightSeedEntityExtractor(catalog)
    linked_entities = entity_extractor.extract(question=question, question_type_hint=question_type_hint)
    grounder = seed_grounder or _build_seed_grounder(
        grounding_mode=normalized_mode,
        catalog=catalog,
        llm_model_name=effective_model_name,
        llm_device=llm_device,
    )
    grounding_decisions = grounder.ground(question=question, entities=linked_entities)
    resolved_entities = _apply_grounding_decisions(linked_entities, grounding_decisions)
    subgraph_edges = retriever.kg_extractor.extract_2hop(
        seed_entities=resolved_entities,
        top_k=retriever.config.primekg_top_k,
        relation_filter=retriever.config.relation_filter,
        question_type=question_type,
        use_relation_filter=retriever.config.use_relation_filter,
        relation_filter_overrides=retriever.config.relation_filter_overrides,
    )
    relation_filter_stats = dict(getattr(retriever.kg_extractor, "last_relation_filter_stats", {}) or {})
    pubmed_passages: list[PubMedPassage] = []
    if hasattr(retriever, "pubmed") and getattr(retriever, "pubmed") is not None:
        pubmed_passages = retriever.pubmed.retrieve(question, k=retriever.config.pubmed_top_k)

    candidate_stats = summarize_grounding_candidates(grounding_decisions)
    unresolved_entities = [decision.surface for decision in grounding_decisions if decision.linked_node_id is None]
    seed_node_ids = [decision.linked_node_id for decision in grounding_decisions if decision.linked_node_id is not None]
    backend_mapping = {
        "entity_linker": getattr(entity_extractor, "backend_used", "unknown"),
        "primekg": getattr(retriever.kg_extractor, "backend_used", "unknown"),
        "pubmed": getattr(getattr(retriever, "pubmed", None), "backend_used", "unknown"),
        "seed_grounding": getattr(grounder, "backend_used", normalized_mode),
    }
    metadata = {
        "entity_count": len(resolved_entities),
        "edge_count": len(subgraph_edges),
        "pubmed_count": len(pubmed_passages),
        "primekg_loaded": bool(getattr(retriever.kg_extractor, "graph", None)),
        "pubmed_cache_path": str(getattr(retriever.config, "pubmed_cache_path", "")),
        "question_type": question_type,
        "question_type_hint": question_type_hint,
        "entity_linker_backend": getattr(entity_extractor, "backend_used", "unknown"),
        "pubmed_backend": getattr(getattr(retriever, "pubmed", None), "backend_used", "unknown"),
        "relation_filter": relation_filter_stats,
        "backend_used": backend_mapping,
        "grounding_mode": normalized_mode,
        "grounding_backend": getattr(grounder, "backend_used", normalized_mode),
        "entity_grounding": serialize_grounding_decisions(grounding_decisions),
        "unresolved_entities": unresolved_entities,
        "seed_node_ids": seed_node_ids,
        "candidate_stats": candidate_stats,
        "llm_grounding_used": bool(getattr(grounder, "llm_used", False)),
        "llm_grounding_fallback_reason": getattr(grounder, "fallback_reason", None),
    }
    return EvidenceBundle(
        question_text=question,
        question_type=question_type,  # type: ignore[arg-type]
        question_entities=resolved_entities,
        subgraph_edges=subgraph_edges,
        pubmed_passages=pubmed_passages,
        metadata=metadata,
    )


def _build_seed_grounder(
    *,
    grounding_mode: str,
    catalog: PrimeKGNodeCatalog,
    llm_model_name: str,
    llm_device: str,
) -> SeedGrounder:
    if grounding_mode == "deterministic_v2":
        return DeterministicSeedGrounder(catalog)
    if grounding_mode == "llm_assisted_v1":
        return LLMAssistedSeedGrounder(catalog, model_name=llm_model_name, device=llm_device)
    raise ValueError(f"Unsupported grounding mode: {grounding_mode}")


def _apply_grounding_decisions(
    entities: Sequence[QuestionEntity],
    decisions: Sequence[Any],
) -> list[QuestionEntity]:
    decision_lookup = {
        (normalize_basic_text(decision.surface), decision.entity_type): decision
        for decision in decisions
    }
    resolved_entities: list[QuestionEntity] = []
    for entity in entities:
        decision = decision_lookup.get((normalize_basic_text(entity.surface), entity.entity_type))
        resolved_entities.append(
            entity.model_copy(update={"primekg_node_id": getattr(decision, "linked_node_id", None)})
            if decision is not None
            else entity.model_copy(deep=True)
        )
    return resolved_entities


def map_medreason_edges(
    *,
    example: RawMedReasonExample,
    evidence: EvidenceBundle,
    backend: str = "heuristic",
    selector: MedReasonEdgeSelector | None = None,
) -> tuple[list[str], EdgeMappingDiagnostics]:
    backend = backend.lower().strip()
    if backend not in {"direct", "heuristic", "llm"}:
        raise ValueError("edge mapper backend must be one of: direct, heuristic, llm.")

    evidence_edge_ids = {edge.edge_id for edge in evidence.subgraph_edges}
    direct_edge_ids = _dedupe(_canonical_edge_id(edge_id) for edge_id in example.direct_edge_ids)
    direct_in_evidence = [edge_id for edge_id in direct_edge_ids if edge_id in evidence_edge_ids]
    missing_direct = [edge_id for edge_id in direct_edge_ids if edge_id not in evidence_edge_ids]
    heuristic_edge_ids = [] if backend == "direct" else heuristic_map_reasoning_to_edges(example, evidence)
    fallback_edge_ids = _dedupe([*direct_in_evidence, *heuristic_edge_ids])
    llm_rationale = ""

    if backend == "direct":
        selected_edge_ids = direct_in_evidence
        backend_used = "direct"
    elif backend == "llm" and selector is not None:
        prompt = _build_edge_selection_prompt(example=example, evidence=evidence, fallback_edge_ids=fallback_edge_ids)
        selection = selector.evaluate(prompt=prompt, fallback_edge_ids=fallback_edge_ids)
        selected_edge_ids = [
            edge_id
            for edge_id in _dedupe(_canonical_edge_id(edge_id) for edge_id in selection.edge_ids)
            if edge_id in evidence_edge_ids
        ]
        if not selected_edge_ids:
            selected_edge_ids = fallback_edge_ids
        llm_rationale = selection.rationale
        backend_used = f"llm_{selector.backend}"
    else:
        selected_edge_ids = fallback_edge_ids
        backend_used = "heuristic"

    diagnostics = EdgeMappingDiagnostics(
        direct_edge_ids=direct_edge_ids,
        direct_edge_ids_in_evidence=direct_in_evidence,
        missing_direct_edge_ids=missing_direct,
        heuristic_edge_ids=heuristic_edge_ids,
        selected_edge_ids=selected_edge_ids,
        backend_used=backend_used,
        llm_rationale=llm_rationale,
    )
    return selected_edge_ids, diagnostics


def heuristic_map_reasoning_to_edges(
    example: RawMedReasonExample,
    evidence: EvidenceBundle,
    *,
    threshold: float = DEFAULT_HEURISTIC_EDGE_THRESHOLD,
) -> list[str]:
    reasoning_text = _normalize_text(
        " ".join(
            fragment
            for fragment in (
                example.question,
                _flatten_to_text(example.reasoning),
            )
            if fragment
        )
    )
    scored_edges: list[tuple[int, float, str]] = []
    for edge in evidence.subgraph_edges:
        score, first_position = _score_edge_against_text(edge, reasoning_text)
        if score >= threshold:
            scored_edges.append((first_position, -score, edge.edge_id))
    scored_edges.sort(key=lambda item: (item[0], item[1], item[2]))
    return [edge_id for _, _, edge_id in scored_edges]


def _score_edge_against_text(edge: KGEdge, text: str) -> tuple[float, int]:
    edge_id = _normalize_text(edge.edge_id)
    head = _normalize_text(_humanize_identifier(edge.head))
    tail = _normalize_text(_humanize_identifier(edge.tail))
    relation = _normalize_text(edge.display_relation or edge.relation)
    relation_tokens = [token for token in _normalize_text(edge.relation.replace("_", " ")).split() if len(token) > 2]
    first_positions = [len(text)]
    score = 0.0

    if edge_id and edge_id in text:
        score += 100.0
        first_positions.append(text.find(edge_id))
    head_score, head_pos = _phrase_score(head, text)
    tail_score, tail_pos = _phrase_score(tail, text)
    relation_score, relation_pos = _phrase_score(relation, text)
    score += 2.0 * head_score
    score += 2.0 * tail_score
    score += relation_score
    if relation_tokens:
        score += min(sum(0.3 for token in relation_tokens if token in text), 1.2)
    for pmid in edge.supporting_pmids:
        pmid_text = _normalize_text(pmid)
        if pmid_text and pmid_text in text:
            score += 1.0
            first_positions.append(text.find(pmid_text))
    first_positions.extend(position for position in (head_pos, tail_pos, relation_pos) if position >= 0)
    return score, min(first_positions)


def _phrase_score(phrase: str, text: str) -> tuple[float, int]:
    if not phrase:
        return 0.0, -1
    position = text.find(phrase)
    if position >= 0:
        return 1.0, position
    tokens = [token for token in phrase.split() if len(token) > 2]
    if not tokens:
        return 0.0, -1
    matched = [token for token in tokens if token in text]
    if not matched:
        return 0.0, -1
    positions = [text.find(token) for token in matched if text.find(token) >= 0]
    return len(matched) / len(tokens), min(positions) if positions else -1


def _build_edge_selection_prompt(
    *,
    example: RawMedReasonExample,
    evidence: EvidenceBundle,
    fallback_edge_ids: Sequence[str],
) -> str:
    candidate_edges = [
        {
            "edge_id": edge.edge_id,
            "head": edge.head,
            "relation": edge.display_relation or edge.relation,
            "tail": edge.tail,
            "supporting_pmids": edge.supporting_pmids,
        }
        for edge in evidence.subgraph_edges[:DEFAULT_MAX_LLM_CANDIDATE_EDGES]
    ]
    return (
        f"Question:\n{example.question}\n\n"
        f"Gold answer:\n{example.answer}\n\n"
        f"MedReason reasoning:\n{_flatten_to_text(example.reasoning)}\n\n"
        f"Candidate PrimeKG edges:\n{json.dumps(candidate_edges, ensure_ascii=True)}\n\n"
        f"Heuristic fallback edge_ids: {list(fallback_edge_ids)}\n\n"
        "Return the minimal ordered list of candidate edge_ids that support the reasoning chain."
    )


def _load_local_records(source_path: Path, *, split: str | None) -> list[dict[str, Any]]:
    if source_path.is_file():
        records = _load_records_file(source_path, split=None)
        return _apply_default_local_split(records, split=split)
    candidate_files, split_specific = _candidate_files(source_path, split=split)
    if not candidate_files:
        raise FileNotFoundError(f"No MedReason dataset files found under {source_path}.")
    records: list[dict[str, Any]] = []
    for candidate in candidate_files:
        records.extend(_load_records_file(candidate, split=split))
    if not split_specific:
        records = _apply_default_local_split(records, split=split)
    return records


def _candidate_files(source_dir: Path, *, split: str | None) -> tuple[list[Path], bool]:
    candidates: list[Path] = []
    split_specific = False
    if split:
        split_dir = source_dir / split
        if split_dir.exists():
            for extension in SUPPORTED_LOCAL_EXTENSIONS:
                candidates.extend(_filter_dataset_files(sorted(split_dir.glob(f"*{extension}"))))
            split_specific = bool(candidates)
        for extension in SUPPORTED_LOCAL_EXTENSIONS:
            split_matches = _filter_dataset_files(sorted(source_dir.glob(f"{split}*{extension}")))
            candidates.extend(split_matches)
            split_specific = split_specific or bool(split_matches)
    if not candidates:
        for extension in SUPPORTED_LOCAL_EXTENSIONS:
            candidates.extend(_filter_dataset_files(sorted(source_dir.glob(f"*{extension}"))))
    return candidates, split_specific


def _filter_dataset_files(paths: Sequence[Path]) -> list[Path]:
    ignored_names = {
        "config.json",
        "dataset-metadata.json",
        "dataset_info.json",
        "datasets-metadata.json",
        "metadata.json",
        "state.json",
    }
    filtered = [path for path in paths if path.name.lower() not in ignored_names]
    return sorted(filtered, key=_dataset_file_priority)


def _dataset_file_priority(path: Path) -> tuple[int, int, str]:
    suffix_priority = {
        ".jsonl": 0,
        ".parquet": 1,
        ".csv": 2,
        ".json": 3,
    }
    name = path.name.lower()
    stem = path.stem.lower()
    name_priority = 1
    if any(token in stem for token in ("train", "validation", "val", "test", "record", "example", "data")):
        name_priority = 0
    return (suffix_priority.get(path.suffix.lower(), 9), name_priority, name)


def _apply_default_local_split(records: list[dict[str, Any]], *, split: str | None) -> list[dict[str, Any]]:
    if not split:
        return records
    normalized = split.lower().strip()
    if normalized not in {"train", "validation", "val", "test"}:
        return records
    train_end = int(len(records) * DEFAULT_LOCAL_TRAIN_FRACTION)
    validation_end = int(len(records) * (DEFAULT_LOCAL_TRAIN_FRACTION + DEFAULT_LOCAL_VALIDATION_FRACTION))
    if normalized == "train":
        return records[:train_end]
    if normalized in {"validation", "val"}:
        return records[train_end:validation_end]
    return records[validation_end:]


def _load_records_file(path: Path, *, split: str | None) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return [record for record in payload if isinstance(record, dict)]
        if isinstance(payload, dict):
            if split and isinstance(payload.get(split), list):
                return [record for record in payload[split] if isinstance(record, dict)]
            for key in ("train", "validation", "val", "test", "data", "records", "examples"):
                if isinstance(payload.get(key), list):
                    return [record for record in payload[key] if isinstance(record, dict)]
        raise ValueError(f"Could not interpret MedReason JSON structure from {path}.")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("pandas is required to read parquet MedReason files.") from exc
        frame = pd.read_parquet(path)
        return [dict(record) for record in frame.to_dict(orient="records")]
    raise ValueError(f"Unsupported MedReason file extension: {path.suffix}")


def _build_skipped_seed_row(
    *,
    index: int,
    raw_record: Mapping[str, Any],
    example: RawMedReasonExample | None,
    evidence: EvidenceBundle | None,
    diagnostics: EdgeMappingDiagnostics | None,
    skip_stage: str,
    skip_reason: str,
    requested_backend: str,
    effective_backend: str,
    message: str | None = None,
) -> dict[str, Any]:
    evidence_metadata = evidence.metadata if evidence is not None else {}
    linked_entities = [
        entity.model_dump(mode="json") if hasattr(entity, "model_dump") else dict(entity)
        for entity in (evidence.question_entities if evidence is not None else [])
    ]
    edge_mapping_payload = asdict(diagnostics) if diagnostics is not None else {}
    mapped_edge_ids = list(edge_mapping_payload.get("selected_edge_ids") or [])
    missing_gold_edges = list(
        edge_mapping_payload.get("missing_direct_edge_ids")
        or edge_mapping_payload.get("missing_gold_edges")
        or []
    )
    backend_used = str(edge_mapping_payload.get("backend_used") or effective_backend)
    question = example.question if example is not None else _first_text(raw_record, QUESTION_KEYS)
    group_id = example.group_id if example is not None else str(
        raw_record.get("id")
        or raw_record.get("question_id")
        or raw_record.get("uid")
        or raw_record.get("sample_id")
        or index
    )

    row: dict[str, Any] = {
        "index": index,
        "group_id": group_id,
        "question": question,
        "question_type": (
            evidence.question_type
            if evidence is not None
            else infer_question_type(record=raw_record, question=question, fallback_label="other")
        ),
        "backend_used": backend_used,
        "requested_backend": requested_backend,
        "effective_backend": effective_backend,
        "entity_count": len(linked_entities),
        "linked_entities": linked_entities,
        "edge_count": len(evidence.subgraph_edges) if evidence is not None else 0,
        "pubmed_count": len(evidence.pubmed_passages) if evidence is not None else 0,
        "mapped_edge_ids": mapped_edge_ids,
        "missing_gold_edges": missing_gold_edges,
        "retrieval_backend_summary": _retrieval_backend_summary(evidence),
        "grounding_mode": evidence_metadata.get("grounding_mode"),
        "grounding_backend": evidence_metadata.get("grounding_backend"),
        "entity_grounding": evidence_metadata.get("entity_grounding") or [],
        "unresolved_entities": evidence_metadata.get("unresolved_entities") or [],
        "seed_node_ids": evidence_metadata.get("seed_node_ids") or [],
        "candidate_stats": evidence_metadata.get("candidate_stats") or {},
        "llm_grounding_used": bool(evidence_metadata.get("llm_grounding_used")),
        "llm_grounding_fallback_reason": evidence_metadata.get("llm_grounding_fallback_reason"),
        "skip_stage": skip_stage,
        "skip_reason": skip_reason,
        "reason": skip_reason,
        "diagnostics": edge_mapping_payload,
    }
    if example is not None:
        row["answer"] = example.answer
        row["direct_edge_ids"] = list(example.direct_edge_ids)
    if message:
        row["message"] = message
    return row


def _infer_no_gold_skip_stage(evidence: EvidenceBundle) -> str:
    if not evidence.question_entities:
        return "entity_linking"
    if not evidence.subgraph_edges:
        return "retrieval"
    return "edge_mapping"


def _retrieval_backend_summary(evidence: EvidenceBundle | None) -> dict[str, Any]:
    if evidence is None:
        return {}
    backend_used = evidence.metadata.get("backend_used")
    backend_mapping = backend_used if isinstance(backend_used, Mapping) else {}
    return {
        "entity_linker_backend": evidence.metadata.get("entity_linker_backend") or backend_mapping.get("entity_linker"),
        "grounding_backend": evidence.metadata.get("grounding_backend") or backend_mapping.get("seed_grounding"),
        "primekg_backend": evidence.metadata.get("primekg_backend") or backend_mapping.get("primekg"),
        "pubmed_backend": evidence.metadata.get("pubmed_backend") or backend_mapping.get("pubmed"),
        "layer1_backend": dict(backend_mapping) if isinstance(backend_used, Mapping) else backend_used,
        "question_type": evidence.metadata.get("question_type") or evidence.question_type,
        "relation_filter": evidence.metadata.get("relation_filter"),
    }


def _loggable_skip_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "index": row.get("index"),
        "group_id": row.get("group_id"),
        "skip_stage": row.get("skip_stage"),
        "skip_reason": row.get("skip_reason"),
        "reason": row.get("skip_reason"),
        "backend_used": row.get("backend_used"),
        "requested_backend": row.get("requested_backend"),
        "effective_backend": row.get("effective_backend"),
        "question_type": row.get("question_type"),
        "entity_count": row.get("entity_count"),
        "edge_count": row.get("edge_count"),
        "pubmed_count": row.get("pubmed_count"),
        "mapped_edge_count": len(_list_or_empty(row.get("mapped_edge_ids"))),
        "missing_gold_edge_count": len(_list_or_empty(row.get("missing_gold_edges"))),
    }


def _summarize_skipped_rows(skipped_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stage_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    edge_counts: list[int] = []
    entity_counts: list[int] = []
    for row in skipped_rows:
        stage = str(row.get("skip_stage") or "unknown")
        reason = str(row.get("skip_reason") or row.get("reason") or "unknown")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        edge_counts.append(int(row.get("edge_count") or 0))
        entity_counts.append(int(row.get("entity_count") or 0))
    return {
        "skip_stage_counts": stage_counts,
        "skip_reason_counts": reason_counts,
        "skipped_avg_edge_count": _mean_ints(edge_counts),
        "skipped_avg_entity_count": _mean_ints(entity_counts),
    }


def _question_type_hint_from_record(record: Mapping[str, Any], *, question: str) -> str | None:
    hint_fields = ("question_type", "type", "puzzle_id", "puzzle_identifiers")
    if not any(field_name in record for field_name in hint_fields):
        return None
    hinted_question_type = infer_question_type(record=record, question=question, fallback_label="other")
    return hinted_question_type or None


def _list_or_empty(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


def _mean_ints(values: Sequence[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _load_hf_records(source: str, *, split: str | None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to load MedReason from Hugging Face.") from exc
    dataset = load_dataset(source, split=split or "train")
    return [dict(record) for record in dataset]


def _resolve_source_path(source: str | Path) -> Path | None:
    path = Path(source)
    resolved = path if path.is_absolute() else KaggleEnv.path(path)
    return resolved if resolved.exists() else None


def _is_real_llm_model_name(model_name: str) -> bool:
    normalized = model_name.strip().lower()
    return normalized not in {"", "none", "null", "false", DEFAULT_HEURISTIC_SELECTOR_NAME}


def _normalize_llm_model_alias(model_name: str) -> str:
    normalized = model_name.strip()
    alias_map = {
        "4omini": DEFAULT_OPENAI_EDGE_SELECTOR,
        "4o-mini": DEFAULT_OPENAI_EDGE_SELECTOR,
        "gpt4omini": DEFAULT_OPENAI_EDGE_SELECTOR,
        "gpt-4omini": DEFAULT_OPENAI_EDGE_SELECTOR,
    }
    return alias_map.get(normalized.lower(), normalized)


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
        return " ".join(_flatten_to_text(item) for item in value.values() if _flatten_to_text(item))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return " ".join(_flatten_to_text(item) for item in value if _flatten_to_text(item))
    return str(value).strip()


def _options_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return " ".join(f"({key}) {item}" for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return " ".join(f"({index}) {item}" for index, item in enumerate(value))
    return str(value)


def _canonical_edge_id(edge_id: Any) -> str:
    return str(edge_id).strip().removeprefix("edge:").removeprefix("edge_id:")


def _dedupe(items: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        if not item or item in deduped:
            continue
        deduped.append(item)
    return deduped


def _humanize_identifier(identifier: str) -> str:
    normalized = identifier.split(":", 1)[-1]
    return normalized.replace("_", " ").replace("-", " ").strip() or identifier


def _normalize_text(text: str) -> str:
    normalized = text.lower().replace("_", " ").replace("-", " ")
    return " ".join("".join(character if character.isalnum() else " " for character in normalized).split())


__all__ = [
    "DEFAULT_GROUNDING_MODE",
    "DEFAULT_MEDREASON_SOURCE",
    "HF_MEDREASON_SOURCE",
    "GROUNDING_MODES",
    "EdgeMappingDiagnostics",
    "EdgeSelection",
    "MedReasonSeedResult",
    "RawMedReasonExample",
    "build_seed_evidence_bundle",
    "heuristic_map_reasoning_to_edges",
    "load_medreason_records",
    "map_medreason_edges",
    "normalize_medreason_record",
    "prepare_medreason_seed_jsonl",
    "resolve_edge_mapper_config",
    "resolve_grounding_mode_config",
]
