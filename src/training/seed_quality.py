from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utils.kaggle_env import KaggleEnv


DEFAULT_GOLDISH_BACKEND_PATTERNS = ("direct", "llm_*")
DEFAULT_REJECT_PRIMEKG_BACKENDS = frozenset({"empty_graph", "primekg_missing"})
DEFAULT_MIN_GOLDISH_LABELED_TOKEN_RATIO = 0.8
DEFAULT_MIN_GOLDISH_GOLD_NODE_COVERAGE = 0.8
DEFAULT_MIN_GOLDISH_GOLD_EDGE_COVERAGE = 1.0
DEFAULT_COVERAGE_TOP_K = 256


@dataclass(slots=True)
class SeedCoverageMetrics:
    gold_node_coverage: float
    gold_edge_coverage: float
    labeled_token_ratio: float
    labeled_token_count: int
    expected_gold_node_count: int
    covered_gold_node_count: int
    expected_gold_edge_count: int
    covered_gold_edge_count: int
    coverage_top_k: int
    missing_gold_nodes: list[str]
    missing_gold_edges: list[str]
    ppr_pruned_gold_nodes: list[str]
    labeled_gold_nodes: list[str]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SeedQualityDecision:
    tier: str
    reasons: list[str]
    gold_edge_count: int
    evidence_edge_count: int
    gold_node_coverage: float
    gold_edge_coverage: float
    labeled_token_ratio: float
    labeled_token_count: int
    coverage: SeedCoverageMetrics
    edge_mapper_backend: str
    entity_linker_backend: str
    primekg_backend: str

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["coverage"] = self.coverage.to_json_dict()
        return payload


@dataclass(slots=True)
class SeedQualitySummary:
    input_path: Path
    output_dir: Path
    total_records: int
    goldish_records: int
    silver_records: int
    rejected_records: int
    accepted_goldish: int
    demoted_silver: int
    rejected: int
    output_files: dict[str, str]
    tier_reason_counts: dict[str, dict[str, int]]
    top_failure_reasons: list[dict[str, int | str]]
    coverage_metric_means: dict[str, dict[str, float]]

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
    min_goldish_labeled_token_ratio: float = DEFAULT_MIN_GOLDISH_LABELED_TOKEN_RATIO,
    min_goldish_gold_node_coverage: float = DEFAULT_MIN_GOLDISH_GOLD_NODE_COVERAGE,
    min_goldish_gold_edge_coverage: float = DEFAULT_MIN_GOLDISH_GOLD_EDGE_COVERAGE,
    coverage_top_k: int = DEFAULT_COVERAGE_TOP_K,
) -> SeedQualityDecision:
    evidence = payload.get("evidence")
    evidence_mapping = evidence if isinstance(evidence, Mapping) else {}
    evidence_metadata = evidence_mapping.get("metadata")
    evidence_metadata_mapping = evidence_metadata if isinstance(evidence_metadata, Mapping) else {}
    backend_used = evidence_metadata_mapping.get("backend_used")
    backend_mapping = backend_used if isinstance(backend_used, Mapping) else {}
    edge_mapping = _mapping_or_empty(_nested_get(payload, "metadata", "edge_mapping"))

    gold_edge_ids = _canonical_edge_ids(_list_or_empty(payload.get("gold_edge_ids")))
    subgraph_edges = _list_or_empty(evidence_mapping.get("subgraph_edges"))
    edge_mapper_backend = str(edge_mapping.get("backend_used") or "unknown")
    entity_linker_backend = str(evidence_metadata_mapping.get("entity_linker_backend") or "unknown")
    primekg_backend = str(backend_mapping.get("primekg") or evidence_metadata_mapping.get("primekg_backend") or "unknown")

    gold_edge_count = len(gold_edge_ids)
    evidence_edge_count = len(subgraph_edges)
    coverage = compute_seed_coverage(payload, coverage_top_k=coverage_top_k)
    reject_reasons: list[str] = []
    silver_reasons: list[str] = []

    if gold_edge_count < min_gold_edges:
        reject_reasons.append("empty_gold_path")
    if evidence_edge_count == 0:
        reject_reasons.append("missing_evidence_edges")
    if require_real_primekg and primekg_backend in DEFAULT_REJECT_PRIMEKG_BACKENDS:
        reject_reasons.append(f"primekg_backend:{primekg_backend}")
    if coverage.labeled_token_count == 0:
        reject_reasons.append(_coverage_failure_reason(payload=payload, coverage=coverage))

    if reject_reasons:
        return SeedQualityDecision(
            tier="reject",
            reasons=_dedupe_strings(reject_reasons),
            gold_edge_count=gold_edge_count,
            evidence_edge_count=evidence_edge_count,
            gold_node_coverage=coverage.gold_node_coverage,
            gold_edge_coverage=coverage.gold_edge_coverage,
            labeled_token_ratio=coverage.labeled_token_ratio,
            labeled_token_count=coverage.labeled_token_count,
            coverage=coverage,
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
    if coverage.gold_edge_coverage < min_goldish_gold_edge_coverage:
        silver_reasons.append(_coverage_failure_reason(payload=payload, coverage=coverage))
        silver_reasons.append("gold_edge_coverage_below_threshold")
    if coverage.gold_node_coverage < min_goldish_gold_node_coverage:
        silver_reasons.append(_coverage_failure_reason(payload=payload, coverage=coverage))
        silver_reasons.append("gold_node_coverage_below_threshold")
    if coverage.labeled_token_ratio < min_goldish_labeled_token_ratio:
        silver_reasons.append(_coverage_failure_reason(payload=payload, coverage=coverage))
        silver_reasons.append("labeled_token_ratio_below_threshold")

    return SeedQualityDecision(
        tier="silver" if silver_reasons else "goldish",
        reasons=_dedupe_strings(silver_reasons) or ["passed_goldish_gate"],
        gold_edge_count=gold_edge_count,
        evidence_edge_count=evidence_edge_count,
        gold_node_coverage=coverage.gold_node_coverage,
        gold_edge_coverage=coverage.gold_edge_coverage,
        labeled_token_ratio=coverage.labeled_token_ratio,
        labeled_token_count=coverage.labeled_token_count,
        coverage=coverage,
        edge_mapper_backend=edge_mapper_backend,
        entity_linker_backend=entity_linker_backend,
        primekg_backend=primekg_backend,
    )


def compute_seed_coverage(payload: Mapping[str, Any], *, coverage_top_k: int = DEFAULT_COVERAGE_TOP_K) -> SeedCoverageMetrics:
    evidence = payload.get("evidence")
    evidence_mapping = evidence if isinstance(evidence, Mapping) else {}
    edge_mapping = _mapping_or_empty(_nested_get(payload, "metadata", "edge_mapping"))
    evidence_edges = _edge_lookup(_list_or_empty(evidence_mapping.get("subgraph_edges")))
    evidence_node_ids = _evidence_node_ids(evidence_edges.values())
    payload_gold_edge_ids = _canonical_edge_ids(_list_or_empty(payload.get("gold_edge_ids")))
    direct_gold_edge_ids = _canonical_edge_ids(_list_or_empty(edge_mapping.get("direct_edge_ids")))
    expected_gold_edge_ids = direct_gold_edge_ids or payload_gold_edge_ids
    covered_gold_edge_ids = [edge_id for edge_id in expected_gold_edge_ids if edge_id in evidence_edges]
    missing_gold_edges = [edge_id for edge_id in expected_gold_edge_ids if edge_id not in evidence_edges]

    explicit_gold_nodes = _first_non_empty_sequence(
        payload.get("gold_node_ids"),
        payload.get("gold_path_nodes"),
        _nested_get(payload, "metadata", "gold_node_ids"),
        _nested_get(payload, "metadata", "gold_path_nodes"),
    )
    expected_gold_nodes = _dedupe_strings(str(node_id) for node_id in explicit_gold_nodes)
    if not expected_gold_nodes:
        expected_gold_nodes = _edge_ids_to_ordered_node_path(covered_gold_edge_ids, evidence_edges)

    gold_nodes_in_evidence = [node_id for node_id in expected_gold_nodes if node_id in evidence_node_ids]
    missing_gold_nodes = [node_id for node_id in expected_gold_nodes if node_id not in evidence_node_ids]
    selected_node_ids = _estimate_selected_node_ids(evidence_mapping, coverage_top_k=coverage_top_k)
    labeled_gold_nodes = [node_id for node_id in expected_gold_nodes if node_id in selected_node_ids]
    ppr_pruned_gold_nodes = [node_id for node_id in gold_nodes_in_evidence if node_id not in selected_node_ids]

    expected_gold_edge_count = len(expected_gold_edge_ids)
    expected_gold_node_count = len(expected_gold_nodes)
    gold_edge_coverage = len(covered_gold_edge_ids) / expected_gold_edge_count if expected_gold_edge_count else 0.0
    gold_node_coverage = len(gold_nodes_in_evidence) / expected_gold_node_count if expected_gold_node_count else 0.0
    labeled_token_ratio = len(labeled_gold_nodes) / expected_gold_node_count if expected_gold_node_count else 0.0

    return SeedCoverageMetrics(
        gold_node_coverage=float(gold_node_coverage),
        gold_edge_coverage=float(gold_edge_coverage),
        labeled_token_ratio=float(labeled_token_ratio),
        labeled_token_count=len(labeled_gold_nodes),
        expected_gold_node_count=expected_gold_node_count,
        covered_gold_node_count=len(gold_nodes_in_evidence),
        expected_gold_edge_count=expected_gold_edge_count,
        covered_gold_edge_count=len(covered_gold_edge_ids),
        coverage_top_k=int(max(coverage_top_k, 0)),
        missing_gold_nodes=missing_gold_nodes,
        missing_gold_edges=missing_gold_edges,
        ppr_pruned_gold_nodes=ppr_pruned_gold_nodes,
        labeled_gold_nodes=labeled_gold_nodes,
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
    min_goldish_labeled_token_ratio: float = DEFAULT_MIN_GOLDISH_LABELED_TOKEN_RATIO,
    min_goldish_gold_node_coverage: float = DEFAULT_MIN_GOLDISH_GOLD_NODE_COVERAGE,
    min_goldish_gold_edge_coverage: float = DEFAULT_MIN_GOLDISH_GOLD_EDGE_COVERAGE,
    coverage_top_k: int = DEFAULT_COVERAGE_TOP_K,
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
    coverage_sums: dict[str, Counter[str]] = defaultdict(Counter)
    coverage_counts = Counter()

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
                    min_goldish_labeled_token_ratio=min_goldish_labeled_token_ratio,
                    min_goldish_gold_node_coverage=min_goldish_gold_node_coverage,
                    min_goldish_gold_edge_coverage=min_goldish_gold_edge_coverage,
                    coverage_top_k=coverage_top_k,
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
                    gold_node_coverage=0.0,
                    gold_edge_coverage=0.0,
                    labeled_token_ratio=0.0,
                    labeled_token_count=0,
                    coverage=SeedCoverageMetrics(
                        gold_node_coverage=0.0,
                        gold_edge_coverage=0.0,
                        labeled_token_ratio=0.0,
                        labeled_token_count=0,
                        expected_gold_node_count=0,
                        covered_gold_node_count=0,
                        expected_gold_edge_count=0,
                        covered_gold_edge_count=0,
                        coverage_top_k=coverage_top_k,
                        missing_gold_nodes=[],
                        missing_gold_edges=[],
                        ppr_pruned_gold_nodes=[],
                        labeled_gold_nodes=[],
                    ),
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
            _add_coverage_sums(coverage_sums, coverage_counts, "all", decision)
            _add_coverage_sums(coverage_sums, coverage_counts, decision.tier, decision)
            handles[decision.tier].write(json.dumps(enriched_payload, ensure_ascii=True, sort_keys=True) + "\n")

    top_failure_reasons = _top_failure_reasons(reason_counts)
    summary = SeedQualitySummary(
        input_path=resolved_input,
        output_dir=resolved_output_dir,
        total_records=sum(tier_counts.values()),
        goldish_records=tier_counts["goldish"],
        silver_records=tier_counts["silver"],
        rejected_records=tier_counts["reject"],
        accepted_goldish=tier_counts["goldish"],
        demoted_silver=tier_counts["silver"],
        rejected=tier_counts["reject"],
        output_files={tier: str(path) for tier, path in tier_paths.items()},
        tier_reason_counts={tier: dict(counter) for tier, counter in reason_counts.items()},
        top_failure_reasons=top_failure_reasons,
        coverage_metric_means=_coverage_means(coverage_sums, coverage_counts),
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


def _first_non_empty_sequence(*values: Any) -> list[Any]:
    for value in values:
        items = _list_or_empty(value)
        if items:
            return items
    return []


def _canonical_edge_ids(values: Sequence[Any]) -> list[str]:
    return _dedupe_strings(str(value).strip().removeprefix("edge:").removeprefix("edge_id:") for value in values)


def _edge_lookup(edges: Sequence[Any]) -> dict[str, Mapping[str, Any]]:
    lookup: dict[str, Mapping[str, Any]] = {}
    for edge in edges:
        if not isinstance(edge, Mapping):
            continue
        edge_id = str(edge.get("edge_id") or "").strip().removeprefix("edge:").removeprefix("edge_id:")
        if edge_id:
            lookup[edge_id] = edge
    return lookup


def _evidence_node_ids(edges: Sequence[Mapping[str, Any]]) -> set[str]:
    node_ids: set[str] = set()
    for edge in edges:
        head = edge.get("head")
        tail = edge.get("tail")
        if head:
            node_ids.add(str(head))
        if tail:
            node_ids.add(str(tail))
    return node_ids


def _edge_ids_to_ordered_node_path(
    gold_edge_ids: Sequence[str],
    edge_lookup: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    node_path: list[str] = []
    ordered_edges = [edge_lookup[edge_id] for edge_id in gold_edge_ids if edge_id in edge_lookup]
    for edge_index, edge in enumerate(ordered_edges):
        head = str(edge.get("head") or "")
        tail = str(edge.get("tail") or "")
        if not head or not tail:
            continue
        if not node_path:
            node_path.extend(_orient_initial_edge(head, tail, ordered_edges[edge_index + 1] if edge_index + 1 < len(ordered_edges) else None))
            continue
        if node_path[-1] == head:
            _append_if_not_repeated(node_path, tail)
        elif node_path[-1] == tail:
            _append_if_not_repeated(node_path, head)
        else:
            _append_if_not_repeated(node_path, head)
            _append_if_not_repeated(node_path, tail)
    return node_path


def _orient_initial_edge(head: str, tail: str, next_edge: Mapping[str, Any] | None) -> list[str]:
    if next_edge is None:
        return [head, tail]
    next_nodes = {str(next_edge.get("head") or ""), str(next_edge.get("tail") or "")}
    if head in next_nodes and tail not in next_nodes:
        return [tail, head]
    return [head, tail]


def _append_if_not_repeated(node_path: list[str], node_id: str) -> None:
    if node_id and (not node_path or node_path[-1] != node_id):
        node_path.append(node_id)


def _estimate_selected_node_ids(evidence_mapping: Mapping[str, Any], *, coverage_top_k: int) -> set[str]:
    if coverage_top_k <= 0:
        return set()
    evidence_metadata = evidence_mapping.get("metadata")
    evidence_metadata_mapping = evidence_metadata if isinstance(evidence_metadata, Mapping) else {}
    explicit_selected = _selected_nodes_from_metadata(evidence_metadata_mapping)
    if explicit_selected:
        return {str(node_id) for node_id in explicit_selected[:coverage_top_k]}

    edges = list(_edge_lookup(_list_or_empty(evidence_mapping.get("subgraph_edges"))).values())
    node_ids = sorted(_evidence_node_ids(edges))
    if not node_ids:
        return set()
    seed_node_ids = {
        str(entity.get("primekg_node_id"))
        for entity in _list_or_empty(evidence_mapping.get("question_entities"))
        if isinstance(entity, Mapping) and entity.get("primekg_node_id")
    }
    scores = _pagerank_scores(edges=edges, node_ids=node_ids, seed_node_ids=seed_node_ids)
    ranked_nodes = sorted(node_ids, key=lambda node_id: (-scores.get(node_id, 0.0), node_id))
    return set(ranked_nodes[:coverage_top_k])


def _selected_nodes_from_metadata(metadata: Mapping[str, Any]) -> list[Any]:
    selected_node_ids = _list_or_empty(metadata.get("selected_node_ids"))
    if selected_node_ids:
        return selected_node_ids
    node_mapping = metadata.get("node_mapping")
    if not isinstance(node_mapping, Mapping):
        return []
    indexed_nodes: list[tuple[int, Any]] = []
    for key, node_id in node_mapping.items():
        try:
            position = int(key)
        except (TypeError, ValueError):
            continue
        indexed_nodes.append((position, node_id))
    return [node_id for _position, node_id in sorted(indexed_nodes)]


def _pagerank_scores(
    *,
    edges: Sequence[Mapping[str, Any]],
    node_ids: Sequence[str],
    seed_node_ids: set[str],
) -> dict[str, float]:
    try:
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_nodes_from(node_ids)
        for edge in edges:
            head = str(edge.get("head") or "")
            tail = str(edge.get("tail") or "")
            if head and tail:
                graph.add_edge(head, tail)
        if not graph.edges:
            return {node_id: (1.0 if node_id in seed_node_ids else 0.0) for node_id in node_ids}
        personalization = {node_id: 0.0 for node_id in graph.nodes}
        active_seeds = [node_id for node_id in seed_node_ids if node_id in personalization] or list(node_ids)
        restart_probability = 1.0 / max(len(active_seeds), 1)
        for node_id in active_seeds:
            personalization[node_id] = restart_probability
        return {str(node_id): float(score) for node_id, score in nx.pagerank(graph, alpha=0.85, personalization=personalization).items()}
    except Exception:
        return {node_id: (1.0 if node_id in seed_node_ids else 0.0) for node_id in node_ids}


def _coverage_failure_reason(*, payload: Mapping[str, Any], coverage: SeedCoverageMetrics) -> str:
    if coverage.expected_gold_edge_count == 0:
        return "empty_gold_path"
    if coverage.missing_gold_edges:
        return "missing_edge_mapping"
    if coverage.ppr_pruned_gold_nodes:
        return "missing_due_to_ppr_pruning"
    if coverage.missing_gold_nodes:
        return "missing_entity_link"

    evidence = payload.get("evidence")
    evidence_mapping = evidence if isinstance(evidence, Mapping) else {}
    seed_node_ids = [
        entity.get("primekg_node_id")
        for entity in _list_or_empty(evidence_mapping.get("question_entities"))
        if isinstance(entity, Mapping) and entity.get("primekg_node_id")
    ]
    if not seed_node_ids:
        return "missing_entity_link"
    return "insufficient_gold_path_coverage"


def _dedupe_strings(values: Sequence[Any] | Any) -> list[str]:
    deduped: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in deduped:
            continue
        deduped.append(text)
    return deduped


def _add_coverage_sums(
    coverage_sums: dict[str, Counter[str]],
    coverage_counts: Counter[str],
    tier: str,
    decision: SeedQualityDecision,
) -> None:
    coverage_sums[tier].update(
        {
            "gold_node_coverage": decision.gold_node_coverage,
            "gold_edge_coverage": decision.gold_edge_coverage,
            "labeled_token_ratio": decision.labeled_token_ratio,
            "labeled_token_count": decision.labeled_token_count,
        }
    )
    coverage_counts[tier] += 1


def _coverage_means(
    coverage_sums: Mapping[str, Counter[str]],
    coverage_counts: Mapping[str, int],
) -> dict[str, dict[str, float]]:
    means: dict[str, dict[str, float]] = {}
    for tier, sums in coverage_sums.items():
        count = int(coverage_counts.get(tier, 0))
        if count <= 0:
            continue
        means[tier] = {metric: float(value) / count for metric, value in sums.items()}
    return means


def _top_failure_reasons(reason_counts: Mapping[str, Counter[str]], *, limit: int = 10) -> list[dict[str, int | str]]:
    combined: Counter[str] = Counter()
    for tier, counter in reason_counts.items():
        if tier == "goldish":
            continue
        combined.update({reason: count for reason, count in counter.items() if reason != "passed_goldish_gate"})
    return [
        {"reason": reason, "count": int(count)}
        for reason, count in sorted(combined.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _matches_any_pattern(value: str, patterns: Sequence[str]) -> bool:
    return any(_matches_pattern(value, pattern) for pattern in patterns)


def _matches_pattern(value: str, pattern: str) -> bool:
    normalized_value = value.strip().lower()
    normalized_pattern = pattern.strip().lower()
    if normalized_pattern.endswith("*"):
        return normalized_value.startswith(normalized_pattern[:-1])
    return normalized_value == normalized_pattern


__all__ = [
    "DEFAULT_COVERAGE_TOP_K",
    "DEFAULT_GOLDISH_BACKEND_PATTERNS",
    "DEFAULT_MIN_GOLDISH_GOLD_EDGE_COVERAGE",
    "DEFAULT_MIN_GOLDISH_GOLD_NODE_COVERAGE",
    "DEFAULT_MIN_GOLDISH_LABELED_TOKEN_RATIO",
    "SeedCoverageMetrics",
    "SeedQualityDecision",
    "SeedQualitySummary",
    "classify_seed_record",
    "compute_seed_coverage",
    "filter_seed_jsonl",
]
