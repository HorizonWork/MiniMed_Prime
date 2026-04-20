#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.layers.layer1_retrieval import (  # noqa: E402
    AgenticRetriever,
    DEFAULT_PRIMEKG_TOP_K,
    QUESTION_TYPE_PATTERNS,
    QUESTION_TYPE_RELATION_FILTERS,
)
from src.layers.layer2_embedder import MedicalGraphEmbedder  # noqa: E402
from src.schemas import EvidenceBundle, KGEdge, PubMedPassage, QuestionEntity  # noqa: E402
from src.training.medreason_adapter import (  # noqa: E402
    DEFAULT_HEURISTIC_SELECTOR_NAME,
    DEFAULT_MEDREASON_SOURCE,
    EdgeMappingDiagnostics,
    MedReasonEdgeSelector,
    RawMedReasonExample,
    load_medreason_records,
    map_medreason_edges,
    normalize_medreason_record,
    resolve_edge_mapper_config,
)
from src.training.seed_quality import (  # noqa: E402
    DEFAULT_COVERAGE_TOP_K,
    SeedQualityDecision,
    classify_seed_record,
)
from src.training.trm_dataset_builder import path_to_token_sequence  # noqa: E402
from src.models.trm_wrapper import DEFAULT_IGNORE_LABEL_ID, DEFAULT_TRM_SEQ_LEN  # noqa: E402
from src.utils.kaggle_env import KaggleEnv, T4Hardening  # noqa: E402


STAGE_ORDER = (
    "sample_load",
    "question_type",
    "entity_extraction",
    "primekg_retrieval",
    "pubmed_retrieval",
    "gold_mapping",
    "tensorization_preview",
    "seed_quality",
    "posthoc_audit",
    "final_trace_verdict",
)


@dataclass(slots=True)
class TraceConfig:
    source: str | Path = DEFAULT_MEDREASON_SOURCE
    split: str | None = "train"
    sample_index: int | None = None
    question_id: str | None = None
    random_one: bool = False
    seed: int = 0
    output_dir: Path = Path("artifacts/traces")
    edge_mapper_backend: str = "auto"
    llm_model_name: str = DEFAULT_HEURISTIC_SELECTOR_NAME
    llm_device: str = "cpu"
    primekg_path: Path = Path("data/kg/primekg")
    pubmed_cache_path: Path = Path("data/pubmed_cache.jsonl")
    pubmed_api_key: str | None = None
    relation_filter: set[str] | None = None
    device: str = "cpu"
    max_len: int = DEFAULT_TRM_SEQ_LEN
    coverage_top_k: int = DEFAULT_COVERAGE_TOP_K
    top_edges: int = 25
    save_subgraph_html: bool = False
    run_posthoc_audit: bool = False
    posthoc_audit_mode: str = "answer_only"
    posthoc_audit_model: str = "heuristic"
    posthoc_audit_device: str = "cpu"


def trace_one_sample(
    config: TraceConfig,
    *,
    retriever: AgenticRetriever | None = None,
    embedder: Any | None = None,
) -> dict[str, Any]:
    trace = _new_trace(config)
    output_dir = _resolve_output_dir(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    raw_record: Mapping[str, Any] = {}
    example: RawMedReasonExample | None = None
    selected_index = -1
    sample_id = "sample-load-failed"

    started = time.perf_counter()
    try:
        records = load_medreason_records(config.source, split=config.split)
        selected_index, raw_record = _select_record(records, config)
        example = normalize_medreason_record(raw_record, index=selected_index)
        sample_id = _safe_sample_id(example.group_id)
        trace.update(
            {
                "sample_id": example.group_id,
                "question": example.question,
                "gold_answer": example.answer,
                "gold_reasoning": example.reasoning,
                "record_index": selected_index,
            }
        )
        trace["sample_load"] = _stage(
            started,
            "ok",
            {
                "source": str(config.source),
                "split": config.split,
                "record_count": len(records),
                "record_index": selected_index,
                "group_id": example.group_id,
                "question": example.question,
                "options": raw_record.get("options", raw_record.get("choices")),
                "gold_answer": example.answer,
                "gold_reasoning": example.reasoning,
                "sample_metadata": _sample_metadata(raw_record),
            },
        )
    except Exception as exc:
        trace["sample_load"] = _stage(started, "fail", {"error_type": type(exc).__name__}, exc)
        trace["final_trace_verdict"] = _final_verdict(trace, failed_stage="sample_load", reasons=[str(exc)])
        _write_trace_artifacts(trace, output_dir=output_dir, sample_id=sample_id, save_subgraph_html=config.save_subgraph_html)
        return trace

    if retriever is None:
        retriever = _build_retriever(config)

    question_type = "other"
    question_entities: list[QuestionEntity] = []
    subgraph_edges: list[KGEdge] = []
    pubmed_passages: list[PubMedPassage] = []
    evidence: EvidenceBundle | None = None
    mapped_edge_ids: list[str] = []
    edge_diagnostics: EdgeMappingDiagnostics | None = None
    quality_decision: SeedQualityDecision | None = None

    started = time.perf_counter()
    try:
        question_type = retriever._classify_question_type(example.question)
        trace["question_type"] = _stage(
            started,
            "ok",
            {
                "predicted": question_type,
                "backend": "rule_based",
                "heuristic_path": _matched_question_type_patterns(example.question, question_type),
                "confidence": None,
                "notes": [],
            },
        )
    except Exception as exc:
        trace["question_type"] = _stage(started, "fail", {"predicted": "other", "backend": "rule_based"}, exc)
        question_type = "other"

    started = time.perf_counter()
    try:
        linked_entities = retriever.linker.link(example.question)
        question_entities = retriever.kg_extractor.resolve_seed_entities(linked_entities)
        missing_entities = [entity.surface for entity in question_entities if not entity.primekg_node_id]
        status = "ok" if question_entities and not missing_entities else "warn"
        if not question_entities:
            status = "fail"
        trace["entity_extraction"] = _stage(
            started,
            status,
            {
                "entities": [
                    _entity_trace(entity, backend=getattr(retriever.linker, "backend_used", "unknown"))
                    for entity in question_entities
                ],
                "missing_entities": missing_entities,
                "missing_critical_entities": missing_entities,
                "backend_used": getattr(retriever.linker, "backend_used", "unknown"),
                "entity_count": len(question_entities),
            },
        )
    except Exception as exc:
        trace["entity_extraction"] = _stage(started, "fail", {"entities": [], "missing_entities": []}, exc)
        question_entities = []

    relation_filter = _effective_relation_filter(config, question_type)
    started = time.perf_counter()
    try:
        retrieval_preview = _primekg_retrieval_preview(
            retriever=retriever,
            seed_entities=question_entities,
            relation_filter=relation_filter,
        )
        subgraph_edges = retriever.kg_extractor.extract_2hop(
            seed_entities=question_entities,
            top_k=getattr(retriever.config, "primekg_top_k", DEFAULT_PRIMEKG_TOP_K),
            relation_filter=relation_filter,
        )
        retrieved_nodes = _nodes_from_edges(subgraph_edges)
        status = "ok" if subgraph_edges else "fail"
        trace["primekg_retrieval"] = _stage(
            started,
            status,
            {
                "seed_node_ids": [entity.primekg_node_id for entity in question_entities if entity.primekg_node_id],
                "relation_filter": sorted(relation_filter or []),
                "relation_filter_active": bool(relation_filter),
                "hop1_nodes": retrieval_preview["hop1_nodes"],
                "hop1_edges": retrieval_preview["hop1_edges"],
                "hop2_nodes": retrieval_preview["hop2_nodes"],
                "hop2_edges": retrieval_preview["hop2_edges"],
                "candidate_nodes": retrieval_preview["candidate_nodes"],
                "candidate_edges": retrieval_preview["candidate_edges"],
                "retrieved_nodes": len(retrieved_nodes),
                "retrieved_edges": len(subgraph_edges),
                "ppr_topk": int(getattr(retriever.config, "primekg_top_k", DEFAULT_PRIMEKG_TOP_K)),
                "top_edges": [_edge_trace(edge) for edge in subgraph_edges[: config.top_edges]],
                "backend_used": getattr(retriever.kg_extractor, "backend_used", "unknown"),
                "primekg_loaded": bool(getattr(retriever.kg_extractor, "graph", None)),
            },
        )
    except Exception as exc:
        trace["primekg_retrieval"] = _stage(
            started,
            "fail",
            {
                "seed_node_ids": [entity.primekg_node_id for entity in question_entities if entity.primekg_node_id],
                "relation_filter": sorted(relation_filter or []),
                "retrieved_nodes": 0,
                "retrieved_edges": 0,
                "ppr_topk": int(getattr(retriever.config, "primekg_top_k", DEFAULT_PRIMEKG_TOP_K)),
                "top_edges": [],
            },
            exc,
        )
        subgraph_edges = []

    started = time.perf_counter()
    try:
        pubmed_passages = retriever.pubmed.retrieve(example.question, k=getattr(retriever.config, "pubmed_top_k", 5))
        pubmed_trace = dict(getattr(retriever.pubmed, "last_retrieval_trace", {}) or {})
        trace["pubmed_retrieval"] = _stage(
            started,
            "ok" if pubmed_passages else "warn",
            {
                "queries": [example.question],
                "query_used": example.question,
                "backend_used": getattr(retriever.pubmed, "backend_used", "unknown"),
                "candidate_count": int(pubmed_trace.get("candidate_count") or 0),
                "top_pmids": pubmed_trace.get("top_pmids") or [_passage_trace(passage) for passage in pubmed_passages],
                "passages": [_passage_trace(passage, include_text=True) for passage in pubmed_passages],
                "score_weights": pubmed_trace.get("score_weights", {}),
            },
        )
    except Exception as exc:
        trace["pubmed_retrieval"] = _stage(
            started,
            "fail",
            {
                "queries": [example.question],
                "query_used": example.question,
                "top_pmids": [],
                "passages": [],
                "backend_used": getattr(getattr(retriever, "pubmed", None), "backend_used", "unknown"),
            },
            exc,
        )
        pubmed_passages = []

    evidence = _build_evidence_bundle(
        question=example.question,
        question_type=question_type,
        question_entities=question_entities,
        subgraph_edges=subgraph_edges,
        pubmed_passages=pubmed_passages,
        retriever=retriever,
        relation_filter=relation_filter,
    )

    started = time.perf_counter()
    try:
        effective_backend, effective_llm_model_name = resolve_edge_mapper_config(
            config.edge_mapper_backend,
            config.llm_model_name,
        )
        selector = (
            MedReasonEdgeSelector(model_name=effective_llm_model_name, device=config.llm_device)
            if effective_backend == "llm"
            else None
        )
        mapped_edge_ids, edge_diagnostics = map_medreason_edges(
            example=example,
            evidence=evidence,
            backend=effective_backend,
            selector=selector,
        )
        edge_coverage = _direct_edge_coverage(edge_diagnostics)
        status = "ok" if mapped_edge_ids else "fail"
        if mapped_edge_ids and edge_diagnostics.missing_direct_edge_ids:
            status = "warn"
        trace["gold_mapping"] = _stage(
            started,
            status,
            {
                "gold_edge_ids": list(edge_diagnostics.direct_edge_ids),
                "mapped_edge_ids": list(mapped_edge_ids),
                "missing_gold_edge_ids": list(edge_diagnostics.missing_direct_edge_ids),
                "missing_gold_edges": list(edge_diagnostics.missing_direct_edge_ids),
                "edge_coverage": edge_coverage,
                "requested_backend": config.edge_mapper_backend,
                "effective_backend": effective_backend,
                "backend_used": edge_diagnostics.backend_used,
                "fallback_chain": _fallback_chain(config.edge_mapper_backend, effective_backend, edge_diagnostics.backend_used),
                "heuristic_edge_ids": list(edge_diagnostics.heuristic_edge_ids),
                "llm_rationale": edge_diagnostics.llm_rationale,
            },
        )
    except Exception as exc:
        trace["gold_mapping"] = _stage(
            started,
            "fail",
            {
                "gold_edge_ids": list(example.direct_edge_ids),
                "mapped_edge_ids": [],
                "missing_gold_edge_ids": list(example.direct_edge_ids),
                "edge_coverage": 0.0,
                "requested_backend": config.edge_mapper_backend,
                "effective_backend": None,
                "backend_used": None,
                "fallback_chain": [config.edge_mapper_backend],
            },
            exc,
        )
        mapped_edge_ids = []
        edge_diagnostics = None

    started = time.perf_counter()
    try:
        effective_embedder = embedder or _build_embedder(config)
        encoded = effective_embedder(evidence)
        labels, label_diagnostics = path_to_token_sequence(
            mapped_edge_ids,
            encoded,
            evidence,
            max_len=config.max_len,
        )
        inputs = encoded["inputs"]
        pad_id = int(getattr(effective_embedder, "padding_token_id", 0))
        input_values = inputs.detach().cpu() if hasattr(inputs, "detach") else inputs
        label_values = labels.detach().cpu() if hasattr(labels, "detach") else labels
        node_mapping = dict(encoded.get("node_mapping", {}))
        nonpad_inputs = _count_nonpad(input_values, pad_id)
        nonpad_labels = _count_nonpad(label_values, DEFAULT_IGNORE_LABEL_ID)
        trace["tensorization_preview"] = _stage(
            started,
            "ok" if nonpad_labels > 0 else "fail",
            {
                "inputs_shape": list(inputs.shape),
                "labels_shape": [1, int(labels.shape[0])],
                "puzzle_identifier": _first_puzzle_identifier(encoded),
                "token_type_counts": _token_type_counts(node_mapping, pad_id=pad_id),
                "nonpad_input_tokens": nonpad_inputs,
                "nonpad_label_tokens": nonpad_labels,
                "labeled_token_count": int(label_diagnostics.get("labeled_token_count", nonpad_labels)),
                "gold_path_len": int(label_diagnostics.get("gold_path_len", len(mapped_edge_ids))),
                "label_mode": label_diagnostics.get("label_mode"),
                "missing_gold_nodes": list(label_diagnostics.get("missing_gold_nodes", [])),
                "missing_gold_edges": list(label_diagnostics.get("missing_gold_edges", [])),
                "codebook_version": _codebook_version(effective_embedder),
                "checkpoint_used": {
                    "layer2_backend": getattr(effective_embedder, "backend_used", "unknown"),
                    "backend_used": dict(getattr(effective_embedder, "last_backend_used", {}) or {}),
                },
            },
        )
    except Exception as exc:
        trace["tensorization_preview"] = _stage(
            started,
            "fail",
            {
                "inputs_shape": [0, config.max_len],
                "labels_shape": [0, config.max_len],
                "puzzle_identifier": None,
                "token_type_counts": {},
                "nonpad_input_tokens": 0,
                "nonpad_label_tokens": 0,
                "labeled_token_count": 0,
                "codebook_version": None,
            },
            exc,
        )

    started = time.perf_counter()
    try:
        quality_payload = _seed_quality_payload(
            example=example,
            evidence=evidence,
            mapped_edge_ids=mapped_edge_ids,
            diagnostics=edge_diagnostics,
            config=config,
            record_index=selected_index,
        )
        quality_decision = classify_seed_record(quality_payload, coverage_top_k=config.coverage_top_k)
        trace["seed_quality"] = _stage(
            started,
            "ok" if quality_decision.tier != "reject" else "fail",
            {
                "tier": quality_decision.tier,
                "reasons": list(quality_decision.reasons),
                "warnings": [] if quality_decision.tier == "goldish" else list(quality_decision.reasons),
                "accepted": quality_decision.tier != "reject",
                "train_val_export": quality_decision.tier if quality_decision.tier != "reject" else "reject",
                "gold_node_coverage": quality_decision.gold_node_coverage,
                "gold_edge_coverage": quality_decision.gold_edge_coverage,
                "labeled_token_ratio": quality_decision.labeled_token_ratio,
                "labeled_token_count": quality_decision.labeled_token_count,
                "coverage": quality_decision.coverage.to_json_dict(),
                "edge_mapper_backend": quality_decision.edge_mapper_backend,
                "entity_linker_backend": quality_decision.entity_linker_backend,
                "primekg_backend": quality_decision.primekg_backend,
            },
        )
    except Exception as exc:
        trace["seed_quality"] = _stage(
            started,
            "fail",
            {
                "tier": "reject",
                "reasons": [type(exc).__name__],
                "warnings": [str(exc)],
                "accepted": False,
            },
            exc,
        )

    if config.run_posthoc_audit:
        started = time.perf_counter()
        try:
            from judges.posthoc_auditor import PostHocAuditor

            auditor = PostHocAuditor(model_name=config.posthoc_audit_model, device=config.posthoc_audit_device)
            hard_constraints = _default_posthoc_hard_constraints(question_type=question_type, question=example.question)
            audit_result = auditor.evaluate_with_diagnostics(
                question=example.question,
                evidence_bundle=evidence,
                candidate_answer=example.answer or None,
                current_reasoning_output=_flatten_to_text(example.reasoning),
                gold_cot_steps=_extract_steps(example.reasoning),
                hard_constraints=hard_constraints,
                mode=config.posthoc_audit_mode,  # type: ignore[arg-type]
            )
            trace["posthoc_audit"] = _stage(
                started,
                "ok" if audit_result.output.overall_recommendation != "ABSTAIN" else "warn",
                {
                    "auditor_model": audit_result.auditor_model,
                    "parse_success": audit_result.parse_success,
                    "json_retry_count": audit_result.json_retry_count,
                    "latency_ms": audit_result.latency_ms,
                    "mode": config.posthoc_audit_mode,
                    "result": audit_result.output.model_dump(mode="json"),
                },
            )
        except Exception as exc:
            trace["posthoc_audit"] = _stage(
                started,
                "fail",
                {
                    "auditor_model": config.posthoc_audit_model,
                    "parse_success": False,
                    "json_retry_count": 0,
                    "mode": config.posthoc_audit_mode,
                    "result": None,
                },
                exc,
            )

    trace["final_trace_verdict"] = _final_verdict(trace)
    _write_trace_artifacts(trace, output_dir=output_dir, sample_id=sample_id, save_subgraph_html=config.save_subgraph_html)
    return trace


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trace one MedReason sample through retrieval, tensorization, and quality verdict.")
    parser.add_argument("--source", default=DEFAULT_MEDREASON_SOURCE, help="HF dataset name or local MedReason file/dir.")
    parser.add_argument("--split", default="train", help="Dataset split name; use 'none' for unsplit local files.")
    parser.add_argument("--sample-index", type=int, default=None, help="MedReason record index to trace. Defaults to 0.")
    parser.add_argument("--question-id", default=None, help="Stable MedReason id/question_id/sample_id to trace.")
    parser.add_argument("--random-one", action="store_true", help="Select one reproducible random sample.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --random-one.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/traces"))
    parser.add_argument("--edge-mapper", choices=("auto", "direct", "heuristic", "llm"), default="auto")
    parser.add_argument(
        "--llm-model-name",
        default=os.getenv("MEDREASON_EDGE_LLM", DEFAULT_HEURISTIC_SELECTOR_NAME),
        help="Judge/LLM model for --edge-mapper llm or auto.",
    )
    parser.add_argument("--llm-device", default="cpu")
    parser.add_argument("--primekg-path", type=Path, default=Path("data/kg/primekg"))
    parser.add_argument("--pubmed-cache-path", type=Path, default=Path("data/pubmed_cache.jsonl"))
    parser.add_argument("--pubmed-api-key", default=None)
    parser.add_argument("--relation-filter", default=None, help="Comma-separated PrimeKG relations; omit to use question-type filter.")
    parser.add_argument("--device", default="cpu", help="Tensorization device.")
    parser.add_argument("--max-len", type=int, default=DEFAULT_TRM_SEQ_LEN)
    parser.add_argument("--coverage-top-k", type=int, default=DEFAULT_COVERAGE_TOP_K)
    parser.add_argument("--top-edges", type=int, default=25)
    parser.add_argument("--save-subgraph-html", action="store_true", help="Write a simple standalone subgraph HTML preview.")
    parser.add_argument("--run-posthoc-audit", action="store_true", help="Run post-hoc LLM auditor after retrieval/tensorization trace.")
    parser.add_argument("--posthoc-audit-mode", choices=("answer_only", "answer_plus_cot"), default="answer_only")
    parser.add_argument("--posthoc-audit-model", default=os.getenv("POSTHOC_AUDIT_MODEL", "heuristic"))
    parser.add_argument("--posthoc-audit-device", default="cpu")
    parser.add_argument("--strict-exit-code", action="store_true", help="Return non-zero when the trace verdict is reject.")
    parser.add_argument("--kaggle", action="store_true", help="Enable Kaggle path and T4 memory handling.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.kaggle or KaggleEnv.is_kaggle():
        T4Hardening.setup_memory()

    split = None if str(args.split).lower() in {"", "none", "null", "false"} else args.split
    relation_filter = (
        {relation.strip() for relation in args.relation_filter.split(",") if relation.strip()}
        if args.relation_filter
        else None
    )
    config = TraceConfig(
        source=args.source,
        split=split,
        sample_index=args.sample_index,
        question_id=args.question_id,
        random_one=args.random_one,
        seed=args.seed,
        output_dir=args.output_dir,
        edge_mapper_backend=args.edge_mapper,
        llm_model_name=args.llm_model_name,
        llm_device=args.llm_device,
        primekg_path=args.primekg_path,
        pubmed_cache_path=args.pubmed_cache_path,
        pubmed_api_key=args.pubmed_api_key,
        relation_filter=relation_filter,
        device=args.device,
        max_len=args.max_len,
        coverage_top_k=args.coverage_top_k,
        top_edges=args.top_edges,
        save_subgraph_html=args.save_subgraph_html,
        run_posthoc_audit=args.run_posthoc_audit,
        posthoc_audit_mode=args.posthoc_audit_mode,
        posthoc_audit_model=args.posthoc_audit_model,
        posthoc_audit_device=args.posthoc_audit_device,
    )
    trace = trace_one_sample(config)
    _print_terminal_summary(trace)
    verdict = trace.get("final_trace_verdict", {})
    if args.strict_exit_code:
        return 0 if verdict.get("status") in {"golden", "partial"} else 1
    return 0


def _new_trace(config: TraceConfig) -> dict[str, Any]:
    config_payload = asdict(config)
    config_payload["source"] = str(config_payload["source"])
    config_payload["output_dir"] = str(config_payload["output_dir"])
    config_payload["primekg_path"] = str(config_payload["primekg_path"])
    config_payload["pubmed_cache_path"] = str(config_payload["pubmed_cache_path"])
    config_payload["relation_filter"] = sorted(config.relation_filter or [])
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_id": None,
        "question": "",
        "gold_answer": "",
        "gold_reasoning": None,
        "stage_order": list(STAGE_ORDER),
        "config": config_payload,
    }


def _build_retriever(config: TraceConfig) -> AgenticRetriever:
    return AgenticRetriever(
        primekg_path=KaggleEnv.path(config.primekg_path),
        pubmed_api_key=config.pubmed_api_key,
        pubmed_cache_path=KaggleEnv.ensure_writeable(KaggleEnv.path(config.pubmed_cache_path)),
        relation_filter=config.relation_filter,
    )


def _build_embedder(config: TraceConfig) -> MedicalGraphEmbedder:
    return MedicalGraphEmbedder(
        sapbert_model_name=_optional_path("data/checkpoints/sapbert"),
        medcpt_article_model_name=_optional_path("data/checkpoints/medcpt-article"),
        device=config.device,
        max_len=config.max_len,
    )


def _select_record(records: Sequence[Mapping[str, Any]], config: TraceConfig) -> tuple[int, Mapping[str, Any]]:
    if not records:
        raise ValueError("No MedReason records were loaded.")
    selector_count = sum(
        [
            config.sample_index is not None,
            bool(config.question_id),
            bool(config.random_one),
        ]
    )
    if selector_count > 1:
        raise ValueError("Use only one selector: --sample-index, --question-id, or --random-one.")
    if config.question_id:
        for index, record in enumerate(records):
            if _record_identifier(record, index) == config.question_id:
                return index, record
        raise ValueError(f"Question id not found: {config.question_id}")
    if config.random_one:
        index = random.Random(config.seed).randrange(len(records))
        return index, records[index]
    index = int(config.sample_index if config.sample_index is not None else 0)
    if index < 0 or index >= len(records):
        raise IndexError(f"sample_index {index} is outside loaded record range 0..{len(records) - 1}.")
    return index, records[index]


def _record_identifier(record: Mapping[str, Any], index: int) -> str:
    return str(
        record.get("id")
        or record.get("question_id")
        or record.get("uid")
        or record.get("sample_id")
        or index
    )


def _stage(
    started_at: float,
    status: str,
    payload: Mapping[str, Any],
    error: Exception | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "duration_ms": (time.perf_counter() - started_at) * 1000.0,
        "error_message": str(error) if error is not None else None,
        **dict(payload),
    }


def _matched_question_type_patterns(question: str, question_type: str) -> list[str]:
    normalized = question.lower()
    return [pattern for pattern in QUESTION_TYPE_PATTERNS.get(question_type, ()) if pattern in normalized]


def _effective_relation_filter(config: TraceConfig, question_type: str) -> set[str] | None:
    if config.relation_filter is not None:
        return set(config.relation_filter)
    default_filter = QUESTION_TYPE_RELATION_FILTERS.get(question_type)
    return set(default_filter) if default_filter else None


def _entity_trace(entity: QuestionEntity, *, backend: str) -> dict[str, Any]:
    confidence = 0.97 if backend == "scispacy_umls" else 1.0
    return {
        "surface": entity.surface,
        "cui": entity.cui,
        "primekg_node_id": entity.primekg_node_id,
        "entity_type": entity.entity_type,
        "confidence": confidence,
        "backend": backend,
    }


def _primekg_retrieval_preview(
    *,
    retriever: AgenticRetriever,
    seed_entities: Sequence[QuestionEntity],
    relation_filter: set[str] | None,
) -> dict[str, int]:
    graph = getattr(retriever.kg_extractor, "graph", None)
    if not graph:
        return {
            "hop1_nodes": 0,
            "hop1_edges": 0,
            "hop2_nodes": 0,
            "hop2_edges": 0,
            "candidate_nodes": 0,
            "candidate_edges": 0,
        }

    seed_node_ids = {entity.primekg_node_id for entity in seed_entities if entity.primekg_node_id}
    if not seed_node_ids:
        return {
            "hop1_nodes": 0,
            "hop1_edges": 0,
            "hop2_nodes": 0,
            "hop2_edges": 0,
            "candidate_nodes": 0,
            "candidate_edges": 0,
        }
    allowed_relations = {relation.lower() for relation in relation_filter} if relation_filter else None
    hop1_edges: dict[str, KGEdge] = {}
    hop1_nodes: set[str] = set()
    for node_id in seed_node_ids:
        for edge in retriever.kg_extractor._iter_incident_edges(node_id=node_id, allowed_relations=allowed_relations):
            hop1_edges[edge.edge_id] = edge
            hop1_nodes.update((edge.head, edge.tail))

    hop2_edges: dict[str, KGEdge] = {}
    hop2_nodes: set[str] = set()
    for node_id in hop1_nodes - seed_node_ids:
        for edge in retriever.kg_extractor._iter_incident_edges(node_id=node_id, allowed_relations=allowed_relations):
            hop2_edges[edge.edge_id] = edge
            hop2_nodes.update((edge.head, edge.tail))

    candidate_edge_ids = set(hop1_edges) | set(hop2_edges)
    candidate_nodes = set(seed_node_ids) | hop1_nodes | hop2_nodes
    return {
        "hop1_nodes": len(hop1_nodes),
        "hop1_edges": len(hop1_edges),
        "hop2_nodes": len(hop2_nodes - hop1_nodes - seed_node_ids),
        "hop2_edges": len(set(hop2_edges) - set(hop1_edges)),
        "candidate_nodes": len(candidate_nodes),
        "candidate_edges": len(candidate_edge_ids),
    }


def _build_evidence_bundle(
    *,
    question: str,
    question_type: str,
    question_entities: Sequence[QuestionEntity],
    subgraph_edges: Sequence[KGEdge],
    pubmed_passages: Sequence[PubMedPassage],
    retriever: AgenticRetriever,
    relation_filter: set[str] | None,
) -> EvidenceBundle:
    metadata = {
        "question_type": question_type,
        "entity_count": len(question_entities),
        "edge_count": len(subgraph_edges),
        "pubmed_count": len(pubmed_passages),
        "relation_filter": sorted(relation_filter or []),
        "entity_linker_backend": getattr(retriever.linker, "backend_used", "unknown"),
        "primekg_backend": getattr(retriever.kg_extractor, "backend_used", "unknown"),
        "pubmed_backend": getattr(retriever.pubmed, "backend_used", "unknown"),
        "backend_used": {
            "entity_linker": getattr(retriever.linker, "backend_used", "unknown"),
            "primekg": getattr(retriever.kg_extractor, "backend_used", "unknown"),
            "pubmed": getattr(retriever.pubmed, "backend_used", "unknown"),
        },
    }
    return EvidenceBundle(
        question_text=question,
        question_type=question_type,  # type: ignore[arg-type]
        question_entities=list(question_entities),
        subgraph_edges=list(subgraph_edges),
        pubmed_passages=list(pubmed_passages),
        metadata=metadata,
    )


def _seed_quality_payload(
    *,
    example: RawMedReasonExample,
    evidence: EvidenceBundle,
    mapped_edge_ids: Sequence[str],
    diagnostics: EdgeMappingDiagnostics | None,
    config: TraceConfig,
    record_index: int,
) -> dict[str, Any]:
    return {
        "evidence": evidence.model_dump(mode="json"),
        "gold_edge_ids": list(mapped_edge_ids),
        "answer": example.answer,
        "reasoning": example.reasoning,
        "group_id": example.group_id,
        "metadata": {
            "source": str(config.source),
            "split": config.split,
            "record_index": record_index,
            "edge_mapping": asdict(diagnostics) if diagnostics is not None else {},
        },
    }


def _direct_edge_coverage(diagnostics: EdgeMappingDiagnostics) -> float:
    if not diagnostics.direct_edge_ids:
        return 1.0 if diagnostics.selected_edge_ids else 0.0
    return len(diagnostics.direct_edge_ids_in_evidence) / len(diagnostics.direct_edge_ids)


def _fallback_chain(requested_backend: str, effective_backend: str, backend_used: str) -> list[str]:
    chain = [requested_backend]
    if effective_backend not in chain:
        chain.append(effective_backend)
    if backend_used not in chain:
        chain.append(backend_used)
    return chain


def _final_verdict(
    trace: Mapping[str, Any],
    *,
    failed_stage: str | None = None,
    reasons: Sequence[str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    failure_stage = failed_stage or _first_failed_stage(trace)
    collected_reasons = list(reasons or _collect_reasons(trace))
    seed_quality = trace.get("seed_quality", {}) if isinstance(trace.get("seed_quality"), Mapping) else {}
    tensorization = trace.get("tensorization_preview", {}) if isinstance(trace.get("tensorization_preview"), Mapping) else {}
    posthoc = trace.get("posthoc_audit", {}) if isinstance(trace.get("posthoc_audit"), Mapping) else {}
    posthoc_result = posthoc.get("result", {}) if isinstance(posthoc.get("result"), Mapping) else {}
    tier = str(seed_quality.get("tier") or "reject")
    labeled_token_count = int(tensorization.get("labeled_token_count") or seed_quality.get("labeled_token_count") or 0)
    posthoc_label = str(posthoc_result.get("sample_label") or "").strip().lower()
    posthoc_recommendation = str(posthoc_result.get("overall_recommendation") or "").strip().upper()
    posthoc_h5 = bool(posthoc_result.get("h5_present"))

    if posthoc_label == "reject" or posthoc_recommendation == "ABSTAIN" or posthoc_h5:
        status = "reject"
        if failure_stage is None:
            failure_stage = "posthoc_audit"
    elif posthoc_label == "golden" and failure_stage is None and labeled_token_count > 0:
        status = "golden"
    elif posthoc_label == "partial" and failure_stage is None and labeled_token_count > 0:
        status = "partial"
    elif failure_stage is None and tier == "goldish" and labeled_token_count > 0:
        status = "golden"
    elif failure_stage is None and tier == "silver" and labeled_token_count > 0:
        status = "partial"
    elif failure_stage in {"pubmed_retrieval"} and labeled_token_count > 0:
        status = "partial"
    else:
        status = "reject"
        if failure_stage is None:
            failure_stage = "seed_quality"
    return _stage(
        started,
        "ok",
        {
            "status": status,
            "failed_stage": failure_stage,
            "reasons": collected_reasons,
            "posthoc_recommendation": posthoc_recommendation or None,
            "posthoc_sample_label": posthoc_label or None,
            "next_debug_hint": _next_debug_hint(failure_stage, trace),
        },
    )


def _first_failed_stage(trace: Mapping[str, Any]) -> str | None:
    for stage_name in STAGE_ORDER:
        if stage_name == "final_trace_verdict":
            continue
        stage = trace.get(stage_name)
        if isinstance(stage, Mapping) and stage.get("status") == "fail":
            return stage_name
    return None


def _collect_reasons(trace: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    for stage_name in STAGE_ORDER:
        stage = trace.get(stage_name)
        if not isinstance(stage, Mapping):
            continue
        if stage.get("error_message"):
            reasons.append(f"{stage_name}:{stage['error_message']}")
        if stage_name == "gold_mapping" and not stage.get("mapped_edge_ids"):
            reasons.append("no_gold_edges_mapped")
        if stage_name == "seed_quality":
            reasons.extend(str(reason) for reason in stage.get("reasons", []) if reason)
        if stage_name == "posthoc_audit":
            result = stage.get("result")
            if isinstance(result, Mapping):
                reasons.extend(str(ref) for ref in result.get("unsupported_evidence_references", []) if ref)
                if result.get("h5_present"):
                    reasons.append("posthoc_h5_detected")
                reasons.extend(str(gap.get("step")) for gap in result.get("reasoning_gaps", []) if isinstance(gap, Mapping))
    return _dedupe(reasons)


def _next_debug_hint(failed_stage: str | None, trace: Mapping[str, Any]) -> str:
    if failed_stage == "sample_load":
        return "Check --source, --split, sample index, or question id."
    if failed_stage == "entity_extraction":
        return "Inspect linked entity surfaces/CUIs and whether PrimeKG node ids resolved."
    if failed_stage == "primekg_retrieval":
        return "Check PrimeKG path, relation_filter, seed_node_ids, and hop counts."
    if failed_stage == "gold_mapping":
        missing = _mapping_get(trace, "gold_mapping", "missing_gold_edge_ids") or []
        return f"Gold edges did not map to retrieved edges. Missing IDs: {missing[:10]}"
    if failed_stage == "tensorization_preview":
        return "Check node_mapping coverage and whether gold path nodes survived PPR/top-k tensor selection."
    if failed_stage == "seed_quality":
        return "Inspect coverage metrics and threshold reasons before dataset export."
    if failed_stage == "posthoc_audit":
        return "Inspect claim-level verdicts, unsupported references, and reasoning gaps in post-hoc audit output."
    if failed_stage == "pubmed_retrieval":
        return "PubMed retrieval failed or returned no passages; inspect cache/API connectivity."
    return "Trace is usable; inspect warn stages for downgraded quality."


def _write_trace_artifacts(
    trace: dict[str, Any],
    *,
    output_dir: Path,
    sample_id: str,
    save_subgraph_html: bool,
) -> None:
    trace_path = output_dir / f"{sample_id}.trace.json"
    summary_path = output_dir / f"{sample_id}.summary.md"
    subgraph_path = output_dir / f"{sample_id}.subgraph.json"
    pubmed_path = output_dir / f"{sample_id}.pubmed.json"
    html_path = output_dir / f"{sample_id}.subgraph.html"

    trace["artifacts"] = {
        "trace_json": str(trace_path),
        "summary_md": str(summary_path),
        "subgraph_json": str(subgraph_path),
        "pubmed_json": str(pubmed_path),
    }
    if save_subgraph_html:
        trace["artifacts"]["subgraph_html"] = str(html_path)

    trace_path.write_text(json.dumps(_json_ready(trace), indent=2, sort_keys=True), encoding="utf-8")
    summary_path.write_text(_summary_markdown(trace), encoding="utf-8")
    subgraph = _subgraph_artifact(trace)
    subgraph_path.write_text(json.dumps(_json_ready(subgraph), indent=2, sort_keys=True), encoding="utf-8")
    pubmed_path.write_text(json.dumps(_json_ready(_pubmed_artifact(trace)), indent=2, sort_keys=True), encoding="utf-8")
    if save_subgraph_html:
        html_path.write_text(_subgraph_html(subgraph), encoding="utf-8")


def _summary_markdown(trace: Mapping[str, Any]) -> str:
    verdict = trace.get("final_trace_verdict", {}) if isinstance(trace.get("final_trace_verdict"), Mapping) else {}
    lines = [
        f"# Trace {trace.get('sample_id')}",
        "",
        f"Question: {trace.get('question', '')}",
        "",
        f"Verdict: **{verdict.get('status', 'unknown')}**",
        f"Failed stage: `{verdict.get('failed_stage')}`",
        "",
        "| Stage | Status | Duration ms | Key details |",
        "| --- | --- | ---: | --- |",
    ]
    for stage_name in STAGE_ORDER:
        stage = trace.get(stage_name)
        if not isinstance(stage, Mapping):
            continue
        lines.append(
            "| {stage} | {status} | {duration:.1f} | {detail} |".format(
                stage=stage_name,
                status=stage.get("status", "unknown"),
                duration=float(stage.get("duration_ms") or 0.0),
                detail=_stage_detail(stage_name, stage).replace("|", "\\|"),
            )
        )

    tensor = trace.get("tensorization_preview", {}) if isinstance(trace.get("tensorization_preview"), Mapping) else {}
    quality = trace.get("seed_quality", {}) if isinstance(trace.get("seed_quality"), Mapping) else {}
    labeled = int(tensor.get("labeled_token_count") or quality.get("labeled_token_count") or 0)
    ratio = float(quality.get("labeled_token_ratio") or 0.0)
    edge_coverage = float(quality.get("gold_edge_coverage") or _mapping_get(trace, "gold_mapping", "edge_coverage") or 0.0)
    if labeled == 0 or ratio < 0.8:
        lines.extend(
            [
                "",
                "## Critical Label Coverage",
                "",
                (
                    '<span style="color:red">CRITICAL: labeled_token_count='
                    f"{labeled}, labeled_token_ratio={ratio:.3f}, gold_edge_coverage={edge_coverage:.3f}</span>"
                ),
            ]
        )

    gold_mapping = trace.get("gold_mapping", {}) if isinstance(trace.get("gold_mapping"), Mapping) else {}
    missing_edges = gold_mapping.get("missing_gold_edge_ids") or gold_mapping.get("missing_gold_edges") or []
    if missing_edges:
        lines.extend(["", "## Missing Gold Edges", "", ", ".join(str(edge_id) for edge_id in missing_edges[:50])])

    reasons = verdict.get("reasons") or []
    if reasons:
        lines.extend(["", "## Reasons", ""])
        lines.extend(f"- {reason}" for reason in reasons)
    lines.extend(["", f"Next debug hint: {verdict.get('next_debug_hint', '')}", ""])
    return "\n".join(lines)


def _stage_detail(stage_name: str, stage: Mapping[str, Any]) -> str:
    if stage_name == "sample_load":
        return f"record_index={stage.get('record_index')} group_id={stage.get('group_id')}"
    if stage_name == "question_type":
        return f"predicted={stage.get('predicted')} backend={stage.get('backend')}"
    if stage_name == "entity_extraction":
        return f"entities={stage.get('entity_count')} missing={stage.get('missing_entities')}"
    if stage_name == "primekg_retrieval":
        return f"edges={stage.get('retrieved_edges')} nodes={stage.get('retrieved_nodes')} filter={stage.get('relation_filter')}"
    if stage_name == "pubmed_retrieval":
        return f"pmids={len(stage.get('top_pmids') or [])} backend={stage.get('backend_used')}"
    if stage_name == "gold_mapping":
        return f"mapped={len(stage.get('mapped_edge_ids') or [])} coverage={stage.get('edge_coverage')}"
    if stage_name == "tensorization_preview":
        return f"labels={stage.get('labeled_token_count')} inputs={stage.get('nonpad_input_tokens')}"
    if stage_name == "seed_quality":
        return f"tier={stage.get('tier')} accepted={stage.get('accepted')}"
    if stage_name == "posthoc_audit":
        result = stage.get("result")
        if isinstance(result, Mapping):
            return f"recommendation={result.get('overall_recommendation')} label={result.get('sample_label')}"
        return f"parse_success={stage.get('parse_success')}"
    if stage_name == "final_trace_verdict":
        return f"status={stage.get('status')} failed={stage.get('failed_stage')}"
    return ""


def _subgraph_artifact(trace: Mapping[str, Any]) -> dict[str, Any]:
    primekg = trace.get("primekg_retrieval", {}) if isinstance(trace.get("primekg_retrieval"), Mapping) else {}
    edge_rows = list(primekg.get("top_edges") or [])
    nodes: dict[str, dict[str, Any]] = {}
    for edge in edge_rows:
        if not isinstance(edge, Mapping):
            continue
        for key in ("head", "tail"):
            node_id = str(edge.get(key) or "")
            if node_id:
                nodes.setdefault(node_id, {"node_id": node_id})
    return {
        "sample_id": trace.get("sample_id"),
        "question": trace.get("question"),
        "seed_node_ids": primekg.get("seed_node_ids", []),
        "relation_filter": primekg.get("relation_filter", []),
        "nodes": list(nodes.values()),
        "edges": edge_rows,
        "retrieved_edges": primekg.get("retrieved_edges", 0),
        "retrieved_nodes": primekg.get("retrieved_nodes", 0),
    }


def _pubmed_artifact(trace: Mapping[str, Any]) -> dict[str, Any]:
    pubmed = trace.get("pubmed_retrieval", {}) if isinstance(trace.get("pubmed_retrieval"), Mapping) else {}
    return {
        "sample_id": trace.get("sample_id"),
        "question": trace.get("question"),
        "queries": pubmed.get("queries", []),
        "backend_used": pubmed.get("backend_used"),
        "score_weights": pubmed.get("score_weights", {}),
        "top_pmids": pubmed.get("top_pmids", []),
        "passages": pubmed.get("passages", []),
    }


def _subgraph_html(subgraph: Mapping[str, Any]) -> str:
    payload = json.dumps(_json_ready(subgraph), indent=2, sort_keys=True)
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Subgraph trace</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;}pre{white-space:pre-wrap;}</style>"
        "</head><body><h1>Subgraph Trace</h1><pre>"
        + payload.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        + "</pre></body></html>"
    )


def _print_terminal_summary(trace: Mapping[str, Any]) -> None:
    verdict = trace.get("final_trace_verdict", {}) if isinstance(trace.get("final_trace_verdict"), Mapping) else {}
    print(f"Trace sample: {trace.get('sample_id')}")
    print(f"Verdict: {verdict.get('status')} failed_stage={verdict.get('failed_stage')}")
    for stage_name in STAGE_ORDER:
        stage = trace.get(stage_name)
        if not isinstance(stage, Mapping):
            continue
        status = str(stage.get("status", "unknown"))
        print(f"{stage_name:23s} {status:5s} {float(stage.get('duration_ms') or 0.0):8.1f} ms  {_stage_detail(stage_name, stage)}")
    artifacts = trace.get("artifacts", {}) if isinstance(trace.get("artifacts"), Mapping) else {}
    if artifacts:
        print(f"trace_json={artifacts.get('trace_json')}")
        print(f"summary_md={artifacts.get('summary_md')}")


def _edge_trace(edge: KGEdge) -> dict[str, Any]:
    return edge.model_dump(mode="json")


def _passage_trace(passage: PubMedPassage, *, include_text: bool = False) -> dict[str, Any]:
    payload = {
        "pmid": passage.pmid,
        "title": passage.title,
        "relevance_score": passage.relevance_score,
    }
    if include_text:
        payload["abstract"] = passage.abstract
    return payload


def _nodes_from_edges(edges: Sequence[KGEdge]) -> set[str]:
    return {node_id for edge in edges for node_id in (edge.head, edge.tail)}


def _count_nonpad(values: Any, pad_id: int) -> int:
    if hasattr(values, "ne"):
        return int(values.ne(pad_id).sum().item())
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return sum(_count_nonpad(item, pad_id) if isinstance(item, Sequence) else int(item != pad_id) for item in values)
    return int(values != pad_id)


def _first_puzzle_identifier(encoded: Mapping[str, Any]) -> int | None:
    value = encoded.get("puzzle_identifiers")
    if value is None:
        return None
    if hasattr(value, "detach"):
        flattened = value.detach().cpu().reshape(-1)
        return int(flattened[0].item()) if flattened.numel() else None
    if isinstance(value, Sequence) and value:
        return int(value[0])
    return None


def _token_type_counts(node_mapping: Mapping[Any, Any], *, pad_id: int) -> dict[str, int]:
    del pad_id
    counts = {"node": 0, "passage": 0, "pad": 0, "other": 0}
    for value in node_mapping.values():
        text = str(value)
        if text == "__pad__":
            counts["pad"] += 1
        elif text.startswith("PMID:"):
            counts["passage"] += 1
        elif text:
            counts["node"] += 1
        else:
            counts["other"] += 1
    return counts


def _codebook_version(embedder: Any) -> str:
    size = getattr(embedder, "codebook_size", None)
    return f"deterministic_vq:{size}" if size is not None else "unknown"


def _sample_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("source", "split", "type", "category", "question_type"):
        if key in record:
            metadata[key] = record[key]
    return metadata


def _default_posthoc_hard_constraints(*, question_type: str, question: str) -> dict[str, Any]:
    lowered_question = question.lower()
    return {
        "question_type": question_type,
        "dosage_sensitive": bool(question_type == "dosage" or any(term in lowered_question for term in ("dose", "dosage", "mg", "mcg"))),
        "ddi_sensitive": bool(
            question_type == "drug_interaction"
            or any(term in lowered_question for term in ("interaction", "interact", "ddi", "contraindication", "contraindicated"))
        ),
    }


def _flatten_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        return " ".join(_flatten_to_text(item) for item in value.values() if _flatten_to_text(item))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return " ".join(_flatten_to_text(item) for item in value if _flatten_to_text(item))
    return str(value).strip()


def _extract_steps(value: Any) -> list[str]:
    text = _flatten_to_text(value)
    if not text:
        return []
    return [fragment.strip() for fragment in re.split(r"(?<=[.!?])\s+|\n+", text) if fragment.strip()]


def _safe_sample_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())[:120] or "sample"


def _resolve_output_dir(path: Path) -> Path:
    return KaggleEnv.ensure_writeable(path if path.is_absolute() else KaggleEnv.path(path))


def _optional_path(path: str) -> str | None:
    resolved = KaggleEnv.path(path)
    return str(resolved) if resolved.exists() else None


def _mapping_get(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _dedupe(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
