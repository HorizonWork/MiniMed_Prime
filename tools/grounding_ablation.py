#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import (  # noqa: E402
    DeterministicSeedGrounder,
    LLMAssistedSeedGrounder,
    LightweightSeedEntityExtractor,
    PrimeKGNodeCatalog,
)
from src.schemas import EvidenceBundle  # noqa: E402
from src.training.medreason_adapter import (  # noqa: E402
    DEFAULT_GROUNDING_MODE,
    DEFAULT_HEURISTIC_SELECTOR_NAME,
    DEFAULT_MEDREASON_SOURCE,
    MedReasonEdgeSelector,
    build_seed_evidence_bundle,
    load_medreason_records,
    map_medreason_edges,
    normalize_medreason_record,
    resolve_edge_mapper_config,
)
from src.utils.kaggle_env import KaggleEnv  # noqa: E402

if TYPE_CHECKING:
    from src.layers.layer1_retrieval import AgenticRetriever


@dataclass(slots=True)
class GroundingAuditRow:
    group_id: str
    grounding_mode: str
    question: str
    entity_count: int
    linked_entity_count: int
    unresolved_entities: list[str]
    seed_node_ids: list[str]
    edge_count: int
    mapped_edge_count: int
    grounding_backend: str
    llm_grounding_used: bool
    llm_grounding_fallback_reason: str | None
    corrected_aliases: list[str]


@dataclass(slots=True)
class GroundingCoverageRow:
    group_id: str
    grounding_mode: str
    question: str
    entity_count: int
    linked_entity_count: int
    linked_node_ids: list[str]
    unresolved_entities: list[str]
    grounding_backend: str
    llm_grounding_used: bool
    llm_grounding_fallback_reason: str | None


class DisabledPubMedRetriever:
    backend_used = "disabled"

    def retrieve(self, question: str, k: int = 0) -> list[Any]:
        del question, k
        return []


def trace_grounding_sample(
    *,
    source: str | Path = DEFAULT_MEDREASON_SOURCE,
    split: str | None = "train",
    sample_index: int = 0,
    grounding_mode: str = DEFAULT_GROUNDING_MODE,
    edge_mapper_backend: str = "heuristic",
    llm_model_name: str = DEFAULT_HEURISTIC_SELECTOR_NAME,
    llm_device: str = "cpu",
    retriever: AgenticRetriever | None = None,
) -> dict[str, Any]:
    records = load_medreason_records(source, split=split)
    record = records[sample_index]
    example = normalize_medreason_record(record, index=sample_index)
    retriever = retriever or _build_retriever()
    evidence = build_seed_evidence_bundle(
        question=example.question,
        retriever=retriever,
        question_type_hint=None,
        grounding_mode=grounding_mode,
        llm_model_name=llm_model_name,
        llm_device=llm_device,
    )
    edge_mapper_backend, effective_llm_model_name = resolve_edge_mapper_config(edge_mapper_backend, llm_model_name)
    selector = (
        MedReasonEdgeSelector(model_name=effective_llm_model_name, device=llm_device)
        if edge_mapper_backend == "llm"
        else None
    )
    edge_ids, diagnostics = map_medreason_edges(
        example=example,
        evidence=evidence,
        backend=edge_mapper_backend,
        selector=selector,
    )
    return {
        "group_id": example.group_id,
        "grounding_mode": grounding_mode,
        "question": example.question,
        "grounding_metadata": dict(evidence.metadata),
        "question_entities": [entity.model_dump(mode="json") for entity in evidence.question_entities],
        "edge_count": len(evidence.subgraph_edges),
        "mapped_edge_ids": edge_ids,
        "edge_mapping_diagnostics": asdict(diagnostics),
    }


def audit_grounding_records(
    *,
    source: str | Path = DEFAULT_MEDREASON_SOURCE,
    split: str | None = "train",
    limit: int | None = None,
    grounding_mode: str = DEFAULT_GROUNDING_MODE,
    edge_mapper_backend: str = "heuristic",
    llm_model_name: str = DEFAULT_HEURISTIC_SELECTOR_NAME,
    llm_device: str = "cpu",
    retriever: AgenticRetriever | None = None,
) -> dict[str, Any]:
    records = load_medreason_records(source, split=split, limit=limit)
    retriever = retriever or _build_retriever()
    rows: list[GroundingAuditRow] = []
    unresolved_counter: Counter[str] = Counter()
    corrected_alias_counter: Counter[str] = Counter()
    edge_mapper_backend, effective_llm_model_name = resolve_edge_mapper_config(edge_mapper_backend, llm_model_name)
    selector = (
        MedReasonEdgeSelector(model_name=effective_llm_model_name, device=llm_device)
        if edge_mapper_backend == "llm"
        else None
    )

    for index, record in enumerate(records):
        example = normalize_medreason_record(record, index=index)
        evidence = build_seed_evidence_bundle(
            question=example.question,
            retriever=retriever,
            question_type_hint=None,
            grounding_mode=grounding_mode,
            llm_model_name=llm_model_name,
            llm_device=llm_device,
        )
        edge_ids, _ = map_medreason_edges(
            example=example,
            evidence=evidence,
            backend=edge_mapper_backend,
            selector=selector,
        )
        unresolved_entities = list(evidence.metadata.get("unresolved_entities") or [])
        corrected_aliases = _corrected_aliases(evidence)
        unresolved_counter.update(unresolved_entities)
        corrected_alias_counter.update(corrected_aliases)
        rows.append(
            GroundingAuditRow(
                group_id=example.group_id,
                grounding_mode=grounding_mode,
                question=example.question,
                entity_count=len(evidence.question_entities),
                linked_entity_count=sum(1 for entity in evidence.question_entities if entity.primekg_node_id),
                unresolved_entities=unresolved_entities,
                seed_node_ids=list(evidence.metadata.get("seed_node_ids") or []),
                edge_count=len(evidence.subgraph_edges),
                mapped_edge_count=len(edge_ids),
                grounding_backend=str(evidence.metadata.get("grounding_backend") or grounding_mode),
                llm_grounding_used=bool(evidence.metadata.get("llm_grounding_used")),
                llm_grounding_fallback_reason=evidence.metadata.get("llm_grounding_fallback_reason"),
                corrected_aliases=corrected_aliases,
            )
        )

    total_records = len(rows)
    total_entities = sum(row.entity_count for row in rows)
    total_linked_entities = sum(row.linked_entity_count for row in rows)
    records_with_edges = sum(1 for row in rows if row.edge_count > 0)
    records_with_no_gold = sum(1 for row in rows if row.mapped_edge_count == 0)
    return {
        "grounding_mode": grounding_mode,
        "records": [asdict(row) for row in rows],
        "summary": {
            "record_count": total_records,
            "entity_link_rate": round(total_linked_entities / total_entities, 4) if total_entities else 0.0,
            "unresolved_entity_rate": round(
                (total_entities - total_linked_entities) / total_entities,
                4,
            )
            if total_entities
            else 0.0,
            "retrieved_edge_rate": round(records_with_edges / total_records, 4) if total_records else 0.0,
            "no_gold_skip_rate": round(records_with_no_gold / total_records, 4) if total_records else 0.0,
            "top_unresolved_surfaces": unresolved_counter.most_common(10),
            "top_corrected_aliases": corrected_alias_counter.most_common(10),
        },
    }


def run_grounding_ablation(
    *,
    source: str | Path = DEFAULT_MEDREASON_SOURCE,
    split: str | None = "train",
    limit: int | None = None,
    llm_model_name: str = DEFAULT_HEURISTIC_SELECTOR_NAME,
    llm_device: str = "cpu",
    retriever: AgenticRetriever | None = None,
) -> dict[str, Any]:
    retriever = retriever or _build_retriever()
    audits = {
        mode: audit_grounding_records(
            source=source,
            split=split,
            limit=limit,
            grounding_mode=mode,
            retriever=retriever,
            llm_model_name=llm_model_name,
            llm_device=llm_device,
        )
        for mode in ("baseline_current", "deterministic_v2", "llm_assisted_v1")
    }
    disagreements = _disagreement_table(
        audits["deterministic_v2"]["records"],
        audits["llm_assisted_v1"]["records"],
    )
    return {
        "audits": audits,
        "disagreement_table": disagreements,
    }


def audit_grounding_coverage(
    *,
    source: str | Path = DEFAULT_MEDREASON_SOURCE,
    split: str | None = "train",
    limit: int | None = None,
    grounding_mode: str = DEFAULT_GROUNDING_MODE,
    llm_model_name: str = DEFAULT_HEURISTIC_SELECTOR_NAME,
    llm_device: str = "cpu",
    primekg_path: str | Path | None = None,
) -> dict[str, Any]:
    records = load_medreason_records(source, split=split, limit=limit)
    catalog = PrimeKGNodeCatalog.from_primekg_path(primekg_path or KaggleEnv.path("data/kg/primekg"))
    extractor = LightweightSeedEntityExtractor(catalog)
    if grounding_mode == "baseline_current":
        raise ValueError("coverage mode only supports deterministic_v2 or llm_assisted_v1 because it avoids inference retrieval.")
    if grounding_mode == "llm_assisted_v1":
        grounder = LLMAssistedSeedGrounder(catalog, model_name=llm_model_name, device=llm_device)
    else:
        grounder = DeterministicSeedGrounder(catalog)

    rows: list[GroundingCoverageRow] = []
    unresolved_counter: Counter[str] = Counter()
    for index, record in enumerate(records):
        example = normalize_medreason_record(record, index=index)
        entities = extractor.extract(example.question)
        decisions = grounder.ground(example.question, entities)
        linked_node_ids = [decision.linked_node_id for decision in decisions if decision.linked_node_id]
        unresolved_entities = [decision.surface for decision in decisions if decision.linked_node_id is None]
        unresolved_counter.update(unresolved_entities)
        rows.append(
            GroundingCoverageRow(
                group_id=example.group_id,
                grounding_mode=grounding_mode,
                question=example.question,
                entity_count=len(decisions),
                linked_entity_count=len(linked_node_ids),
                linked_node_ids=linked_node_ids,
                unresolved_entities=unresolved_entities,
                grounding_backend=getattr(grounder, "backend_used", grounding_mode),
                llm_grounding_used=bool(getattr(grounder, "llm_used", False)),
                llm_grounding_fallback_reason=getattr(grounder, "fallback_reason", None),
            )
        )

    total_records = len(rows)
    total_entities = sum(row.entity_count for row in rows)
    total_linked_entities = sum(row.linked_entity_count for row in rows)
    records_with_any_link = sum(1 for row in rows if row.linked_entity_count > 0)
    records_all_unresolved = sum(1 for row in rows if row.entity_count > 0 and row.linked_entity_count == 0)
    return {
        "grounding_mode": grounding_mode,
        "rows": [asdict(row) for row in rows],
        "summary": {
            "record_count": total_records,
            "entity_link_rate": round(total_linked_entities / total_entities, 4) if total_entities else 0.0,
            "record_any_link_rate": round(records_with_any_link / total_records, 4) if total_records else 0.0,
            "record_all_unresolved_rate": round(records_all_unresolved / total_records, 4) if total_records else 0.0,
            "avg_linked_entities_per_record": round(total_linked_entities / total_records, 4) if total_records else 0.0,
            "top_unresolved_surfaces": unresolved_counter.most_common(20),
        },
    }


def _build_retriever(*, include_pubmed: bool = False) -> AgenticRetriever:
    from src.layers.layer1_retrieval import AgenticRetriever

    retriever = AgenticRetriever(
        primekg_path=KaggleEnv.path("data/kg/primekg"),
        pubmed_cache_path=KaggleEnv.path("data/pubmed_cache.jsonl"),
    )
    if not include_pubmed:
        retriever.pubmed = DisabledPubMedRetriever()  # type: ignore[assignment]
    return retriever


def _corrected_aliases(evidence: EvidenceBundle) -> list[str]:
    corrected: list[str] = []
    grounding_rows = list(evidence.metadata.get("entity_grounding") or [])
    if not grounding_rows:
        return corrected
    for row in grounding_rows:
        if not isinstance(row, Mapping):
            continue
        surface = str(row.get("surface") or "")
        linked_node_name = str(row.get("linked_node_name") or "")
        if not surface or not linked_node_name:
            continue
        if normalize_surface(surface) != normalize_surface(linked_node_name):
            corrected.append(f"{surface} -> {linked_node_name}")
    return corrected


def _disagreement_table(
    deterministic_rows: Sequence[Mapping[str, Any]],
    llm_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    disagreements: Counter[str] = Counter()
    for deterministic, llm_row in zip(deterministic_rows, llm_rows):
        deterministic_nodes = tuple(deterministic.get("seed_node_ids") or [])
        llm_nodes = tuple(llm_row.get("seed_node_ids") or [])
        if deterministic_nodes != llm_nodes:
            disagreements[f"{deterministic.get('group_id')}::{deterministic_nodes} != {llm_nodes}"] += 1
    return {
        "disagreement_count": sum(disagreements.values()),
        "rows": disagreements.most_common(20),
    }


def normalize_surface(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _write_payload(payload: Mapping[str, Any], output_path: Path | None) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if output_path is None:
        print(serialized)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grounding-only PrimeKG ablation and analysis.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    trace_parser = subparsers.add_parser("trace", help="Trace a single sample grounding run.")
    trace_parser.add_argument("--source", default=DEFAULT_MEDREASON_SOURCE)
    trace_parser.add_argument("--split", default="train")
    trace_parser.add_argument("--sample-index", type=int, default=0)
    trace_parser.add_argument("--grounding-mode", default=DEFAULT_GROUNDING_MODE)
    trace_parser.add_argument("--edge-mapper-backend", default="heuristic")
    trace_parser.add_argument("--llm-model-name", default=DEFAULT_HEURISTIC_SELECTOR_NAME)
    trace_parser.add_argument("--llm-device", default="cpu")
    trace_parser.add_argument("--output-json", type=Path)

    audit_parser = subparsers.add_parser("audit", help="Audit grounding on a batch of MedReason rows.")
    audit_parser.add_argument("--source", default=DEFAULT_MEDREASON_SOURCE)
    audit_parser.add_argument("--split", default="train")
    audit_parser.add_argument("--limit", type=int)
    audit_parser.add_argument("--grounding-mode", default=DEFAULT_GROUNDING_MODE)
    audit_parser.add_argument("--edge-mapper-backend", default="heuristic")
    audit_parser.add_argument("--llm-model-name", default=DEFAULT_HEURISTIC_SELECTOR_NAME)
    audit_parser.add_argument("--llm-device", default="cpu")
    audit_parser.add_argument("--output-json", type=Path)

    coverage_parser = subparsers.add_parser(
        "coverage",
        help="Run lightweight grounding coverage audit without graph extraction.",
    )
    coverage_parser.add_argument("--source", default=DEFAULT_MEDREASON_SOURCE)
    coverage_parser.add_argument("--split", default="train")
    coverage_parser.add_argument("--limit", type=int)
    coverage_parser.add_argument("--grounding-mode", default="deterministic_v2")
    coverage_parser.add_argument("--llm-model-name", default=DEFAULT_HEURISTIC_SELECTOR_NAME)
    coverage_parser.add_argument("--llm-device", default="cpu")
    coverage_parser.add_argument("--primekg-path", default=str(KaggleEnv.path("data/kg/primekg")))
    coverage_parser.add_argument("--output-json", type=Path)

    ablation_parser = subparsers.add_parser("ablation", help="Run 3-way grounding ablation.")
    ablation_parser.add_argument("--source", default=DEFAULT_MEDREASON_SOURCE)
    ablation_parser.add_argument("--split", default="train")
    ablation_parser.add_argument("--limit", type=int)
    ablation_parser.add_argument("--llm-model-name", default=DEFAULT_HEURISTIC_SELECTOR_NAME)
    ablation_parser.add_argument("--llm-device", default="cpu")
    ablation_parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "trace":
        payload = trace_grounding_sample(
            source=args.source,
            split=args.split,
            sample_index=args.sample_index,
            grounding_mode=args.grounding_mode,
            edge_mapper_backend=args.edge_mapper_backend,
            llm_model_name=args.llm_model_name,
            llm_device=args.llm_device,
        )
    elif args.command == "audit":
        payload = audit_grounding_records(
            source=args.source,
            split=args.split,
            limit=args.limit,
            grounding_mode=args.grounding_mode,
            edge_mapper_backend=args.edge_mapper_backend,
            llm_model_name=args.llm_model_name,
            llm_device=args.llm_device,
        )
    elif args.command == "coverage":
        payload = audit_grounding_coverage(
            source=args.source,
            split=args.split,
            limit=args.limit,
            grounding_mode=args.grounding_mode,
            llm_model_name=args.llm_model_name,
            llm_device=args.llm_device,
            primekg_path=args.primekg_path,
        )
    else:
        payload = run_grounding_ablation(
            source=args.source,
            split=args.split,
            limit=args.limit,
            llm_model_name=args.llm_model_name,
            llm_device=args.llm_device,
        )
    _write_payload(payload, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
