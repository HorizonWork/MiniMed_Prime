from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utils.kaggle_env import KaggleEnv


DEFAULT_GOLDISH_BACKEND_PATTERNS = ("direct", "llm_*")
DEFAULT_REJECT_PRIMEKG_BACKENDS = frozenset({"empty_graph", "primekg_missing"})


@dataclass(slots=True)
class SeedQualityDecision:
    tier: str
    reasons: list[str]
    gold_edge_count: int
    evidence_edge_count: int
    edge_mapper_backend: str
    entity_linker_backend: str
    primekg_backend: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SeedQualitySummary:
    input_path: Path
    output_dir: Path
    total_records: int
    goldish_records: int
    silver_records: int
    rejected_records: int
    output_files: dict[str, str]
    tier_reason_counts: dict[str, dict[str, int]]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_path"] = str(self.input_path)
        payload["output_dir"] = str(self.output_dir)
        return payload


def classify_seed_record(
    payload: Mapping[str, Any],
    *,
    min_gold_edges: int = 1,
    accepted_goldish_edge_backends: Sequence[str] = DEFAULT_GOLDISH_BACKEND_PATTERNS,
    require_real_entity_linker: bool = True,
    require_real_primekg: bool = True,
) -> SeedQualityDecision:
    evidence = payload.get("evidence")
    evidence_mapping = evidence if isinstance(evidence, Mapping) else {}
    evidence_metadata = evidence_mapping.get("metadata")
    evidence_metadata_mapping = evidence_metadata if isinstance(evidence_metadata, Mapping) else {}
    backend_used = evidence_metadata_mapping.get("backend_used")
    backend_mapping = backend_used if isinstance(backend_used, Mapping) else {}
    edge_mapping = _mapping_or_empty(_nested_get(payload, "metadata", "edge_mapping"))

    gold_edge_ids = _list_or_empty(payload.get("gold_edge_ids"))
    subgraph_edges = _list_or_empty(evidence_mapping.get("subgraph_edges"))
    edge_mapper_backend = str(edge_mapping.get("backend_used") or "unknown")
    entity_linker_backend = str(evidence_metadata_mapping.get("entity_linker_backend") or "unknown")
    primekg_backend = str(backend_mapping.get("primekg") or evidence_metadata_mapping.get("primekg_backend") or "unknown")

    gold_edge_count = len(gold_edge_ids)
    evidence_edge_count = len(subgraph_edges)
    reject_reasons: list[str] = []
    silver_reasons: list[str] = []

    if gold_edge_count < min_gold_edges:
        reject_reasons.append("missing_gold_edge_ids")
    if evidence_edge_count == 0:
        reject_reasons.append("missing_evidence_edges")
    if require_real_primekg and primekg_backend in DEFAULT_REJECT_PRIMEKG_BACKENDS:
        reject_reasons.append(f"primekg_backend:{primekg_backend}")

    if reject_reasons:
        return SeedQualityDecision(
            tier="reject",
            reasons=reject_reasons,
            gold_edge_count=gold_edge_count,
            evidence_edge_count=evidence_edge_count,
            edge_mapper_backend=edge_mapper_backend,
            entity_linker_backend=entity_linker_backend,
            primekg_backend=primekg_backend,
        )

    if not _matches_any_pattern(edge_mapper_backend, accepted_goldish_edge_backends):
        silver_reasons.append(f"edge_mapper_backend:{edge_mapper_backend}")
    if require_real_entity_linker and entity_linker_backend != "scispacy_umls":
        silver_reasons.append(f"entity_linker_backend:{entity_linker_backend}")
    if require_real_primekg and primekg_backend == "unknown":
        silver_reasons.append("primekg_backend:unknown")

    return SeedQualityDecision(
        tier="silver" if silver_reasons else "goldish",
        reasons=silver_reasons or ["passed_goldish_gate"],
        gold_edge_count=gold_edge_count,
        evidence_edge_count=evidence_edge_count,
        edge_mapper_backend=edge_mapper_backend,
        entity_linker_backend=entity_linker_backend,
        primekg_backend=primekg_backend,
    )


def filter_seed_jsonl(
    *,
    input_jsonl: Path,
    output_dir: Path,
    goldish_name: str = "trm_seed_goldish.jsonl",
    silver_name: str = "trm_seed_silver.jsonl",
    reject_name: str = "trm_seed_rejects.jsonl",
    summary_name: str = "seed_quality_summary.json",
    min_gold_edges: int = 1,
    accepted_goldish_edge_backends: Sequence[str] = DEFAULT_GOLDISH_BACKEND_PATTERNS,
    require_real_entity_linker: bool = True,
    require_real_primekg: bool = True,
) -> SeedQualitySummary:
    resolved_input = input_jsonl if input_jsonl.is_absolute() else KaggleEnv.path(input_jsonl)
    resolved_output_dir = KaggleEnv.ensure_writeable(output_dir if output_dir.is_absolute() else KaggleEnv.path(output_dir))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    tier_paths = {
        "goldish": resolved_output_dir / goldish_name,
        "silver": resolved_output_dir / silver_name,
        "reject": resolved_output_dir / reject_name,
    }
    reason_counts: dict[str, Counter[str]] = defaultdict(Counter)
    tier_counts = Counter({"goldish": 0, "silver": 0, "reject": 0})

    with (
        resolved_input.open("r", encoding="utf-8") as source,
        tier_paths["goldish"].open("w", encoding="utf-8") as goldish_handle,
        tier_paths["silver"].open("w", encoding="utf-8") as silver_handle,
        tier_paths["reject"].open("w", encoding="utf-8") as reject_handle,
    ):
        handles = {
            "goldish": goldish_handle,
            "silver": silver_handle,
            "reject": reject_handle,
        }
        for line_number, raw_line in enumerate(source, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
                if not isinstance(payload, Mapping):
                    raise ValueError("Seed JSONL rows must decode to JSON objects.")
                decision = classify_seed_record(
                    payload,
                    min_gold_edges=min_gold_edges,
                    accepted_goldish_edge_backends=accepted_goldish_edge_backends,
                    require_real_entity_linker=require_real_entity_linker,
                    require_real_primekg=require_real_primekg,
                )
                enriched_payload = {
                    **payload,
                    "quality": decision.to_json_dict(),
                }
            except Exception as exc:
                decision = SeedQualityDecision(
                    tier="reject",
                    reasons=[f"invalid_row:{type(exc).__name__}"],
                    gold_edge_count=0,
                    evidence_edge_count=0,
                    edge_mapper_backend="unknown",
                    entity_linker_backend="unknown",
                    primekg_backend="unknown",
                )
                enriched_payload = {
                    "_line_number": line_number,
                    "_raw_line": stripped,
                    "quality": decision.to_json_dict(),
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }

            tier_counts[decision.tier] += 1
            reason_counts[decision.tier].update(decision.reasons)
            handles[decision.tier].write(json.dumps(enriched_payload, ensure_ascii=True, sort_keys=True) + "\n")

    summary = SeedQualitySummary(
        input_path=resolved_input,
        output_dir=resolved_output_dir,
        total_records=sum(tier_counts.values()),
        goldish_records=tier_counts["goldish"],
        silver_records=tier_counts["silver"],
        rejected_records=tier_counts["reject"],
        output_files={tier: str(path) for tier, path in tier_paths.items()},
        tier_reason_counts={tier: dict(counter) for tier, counter in reason_counts.items()},
    )
    (resolved_output_dir / summary_name).write_text(
        json.dumps(summary.to_json_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _nested_get(payload: Mapping[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list_or_empty(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


def _matches_any_pattern(value: str, patterns: Sequence[str]) -> bool:
    return any(_matches_pattern(value, pattern) for pattern in patterns)


def _matches_pattern(value: str, pattern: str) -> bool:
    normalized_value = value.strip().lower()
    normalized_pattern = pattern.strip().lower()
    if normalized_pattern.endswith("*"):
        return normalized_value.startswith(normalized_pattern[:-1])
    return normalized_value == normalized_pattern


__all__ = [
    "DEFAULT_GOLDISH_BACKEND_PATTERNS",
    "SeedQualityDecision",
    "SeedQualitySummary",
    "classify_seed_record",
    "filter_seed_jsonl",
]
