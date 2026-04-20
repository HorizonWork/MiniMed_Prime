from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from src.layers.layer2_embedder import QUESTION_TYPE_TO_PUZZLE_ID
from src.models.trm_wrapper import DEFAULT_IGNORE_LABEL_ID, DEFAULT_TRM_SEQ_LEN, DEFAULT_TRM_VOCAB_SIZE
from src.schemas import EvidenceBundle, KGEdge
from src.utils.kaggle_env import KaggleEnv
from src.utils.structured_logger import DEFAULT_LOG_DIR, StructuredLogger


DEFAULT_TRM_NUM_PUZZLE_IDENTIFIERS = len(set(QUESTION_TYPE_TO_PUZZLE_ID.values()))
DEFAULT_FALLBACK_PAD_ID = 0
DEFAULT_EDGE_PATTERN = re.compile(
    r"(?:edge_id|edge)\s*[:=]\s*['\"]?([A-Za-z0-9_.:-]+)|\[edge:([^\]]+)\]",
    re.IGNORECASE,
)
DEFAULT_LABEL_MODE = "ordered_node_token_sequence"


@dataclass(slots=True)
class TRMTrainingExample:
    evidence: EvidenceBundle
    gold_edge_ids: list[str]
    group_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TRMDatasetArrays:
    inputs: np.ndarray
    labels: np.ndarray
    puzzle_identifiers: np.ndarray
    puzzle_indices: np.ndarray
    group_indices: np.ndarray
    metadata: dict[str, Any]
    skipped_examples: list[dict[str, Any]] = field(default_factory=list)
    row_metadata: list[dict[str, Any]] = field(default_factory=list)


def parse_reasoning_chain(reasoning: Any) -> list[str]:
    """Extract canonical edge IDs from MedReason-style CoT payloads."""
    edge_ids: list[str] = []

    def add_edge_id(value: Any) -> None:
        if value is None:
            return
        text = str(value).strip()
        if not text:
            return
        text = text.removeprefix("edge:").removeprefix("edge_id:")
        if text and text not in edge_ids:
            edge_ids.append(text)

    if isinstance(reasoning, Mapping):
        for key in ("gold_edge_ids", "edge_ids", "premise_edge_ids"):
            value = reasoning.get(key)
            if isinstance(value, Sequence) and not isinstance(value, str):
                for item in value:
                    add_edge_id(item)
            else:
                add_edge_id(value)
        for key in ("reasoning", "cot", "chain", "claim_text"):
            value = reasoning.get(key)
            if value is not None:
                for edge_id in parse_reasoning_chain(value):
                    add_edge_id(edge_id)
        return edge_ids

    if isinstance(reasoning, Sequence) and not isinstance(reasoning, str):
        for item in reasoning:
            for edge_id in parse_reasoning_chain(item):
                add_edge_id(edge_id)
        return edge_ids

    text = str(reasoning or "")
    for match in DEFAULT_EDGE_PATTERN.finditer(text):
        add_edge_id(match.group(1) or match.group(2))
    return edge_ids


def path_to_token_sequence(
    gold_edge_ids: Sequence[str],
    embedder_output: Mapping[str, Any],
    evidence: EvidenceBundle,
    *,
    max_len: int = DEFAULT_TRM_SEQ_LEN,
    ignore_label_id: int = DEFAULT_IGNORE_LABEL_ID,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Map a gold KG edge path to an ordered TRM target sequence.

    Layer 2 still exposes node provenance by input position, so the builder uses that
    provenance only to translate gold-path nodes into their quantized token IDs. The
    label positions are answer slots ordered by the gold path, not the retrieval/PPR
    positions of the input tokens.
    """
    inputs = _coerce_inputs(embedder_output["inputs"], max_len=max_len)
    node_mapping = _normalize_node_mapping(embedder_output.get("node_mapping", {}))
    token_by_node = _node_tokens_by_id(inputs=inputs, node_mapping=node_mapping)
    edge_lookup = {edge.edge_id: edge for edge in evidence.subgraph_edges}

    labels = torch.full((max_len,), int(ignore_label_id), dtype=torch.long)
    missing_gold_edges: list[str] = []
    missing_gold_nodes: list[str] = []
    labeled_positions: list[int] = []
    source_positions: list[int] = []
    labeled_node_ids: list[str] = []
    truncated_gold_nodes: list[str] = []

    ordered_gold_nodes = _edge_ids_to_ordered_node_path(gold_edge_ids, edge_lookup, missing_gold_edges)
    for node_id in ordered_gold_nodes:
        token_record = token_by_node.get(node_id)
        if token_record is None:
            if node_id not in missing_gold_nodes:
                missing_gold_nodes.append(node_id)
            continue
        if len(labeled_positions) >= max_len:
            truncated_gold_nodes.append(node_id)
            continue
        target_position = len(labeled_positions)
        source_position, token_id = token_record
        labels[target_position] = int(token_id)
        labeled_positions.append(target_position)
        source_positions.append(source_position)
        labeled_node_ids.append(node_id)

    diagnostics = {
        "gold_edge_ids": list(gold_edge_ids),
        "gold_path_len": len(gold_edge_ids),
        "ordered_gold_nodes": ordered_gold_nodes,
        "labeled_node_ids": labeled_node_ids,
        "missing_gold_edges": missing_gold_edges,
        "missing_gold_nodes": missing_gold_nodes,
        "missing_edge_ids": missing_gold_edges,
        "missing_node_ids": missing_gold_nodes,
        "labeled_positions": labeled_positions,
        "source_positions": source_positions,
        "labeled_token_count": len(labeled_positions),
        "label_mode": DEFAULT_LABEL_MODE,
        "truncated_gold_nodes": truncated_gold_nodes,
    }
    return labels, diagnostics


def build_trm_dataset_arrays(
    examples: Iterable[TRMTrainingExample],
    embedder: Any,
    *,
    max_len: int = DEFAULT_TRM_SEQ_LEN,
    vocab_size: int = DEFAULT_TRM_VOCAB_SIZE,
    ignore_label_id: int = DEFAULT_IGNORE_LABEL_ID,
) -> TRMDatasetArrays:
    started_at = time.perf_counter()
    event_logger = StructuredLogger("trm_dataset_builder", DEFAULT_LOG_DIR)
    input_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    puzzle_ids: list[int] = []
    skipped_examples: list[dict[str, Any]] = []
    row_metadata: list[dict[str, Any]] = []
    pad_id = int(getattr(embedder, "padding_token_id", DEFAULT_FALLBACK_PAD_ID))
    total_seen = 0
    edge_counts: list[int] = []
    node_counts: list[int] = []
    passage_counts: list[int] = []
    gold_path_lengths: list[int] = []
    labeled_token_counts: list[int] = []

    for example_index, example in enumerate(examples):
        total_seen += 1
        edge_count = len(example.evidence.subgraph_edges)
        node_count = len({node for edge in example.evidence.subgraph_edges for node in (edge.head, edge.tail)})
        passage_count = len(example.evidence.pubmed_passages)
        edge_counts.append(edge_count)
        node_counts.append(node_count)
        passage_counts.append(passage_count)
        gold_path_lengths.append(len(example.gold_edge_ids))
        if not example.gold_edge_ids:
            skipped_examples.append(
                {
                    "index": example_index,
                    "reason": "missing_gold_path",
                    "edge_count": edge_count,
                    "node_count": node_count,
                    "passage_count": passage_count,
                    "gold_path_len": 0,
                    "labeled_token_count": 0,
                    "label_mode": DEFAULT_LABEL_MODE,
                    "missing_gold_nodes": [],
                    "missing_gold_edges": [],
                }
            )
            continue

        encoded = embedder(example.evidence)
        inputs = _coerce_inputs(encoded["inputs"], max_len=max_len)
        labels, diagnostics = path_to_token_sequence(
            gold_edge_ids=example.gold_edge_ids,
            embedder_output=encoded,
            evidence=example.evidence,
            max_len=max_len,
            ignore_label_id=ignore_label_id,
        )
        if diagnostics["labeled_token_count"] == 0:
            skipped_examples.append(
                {
                    "index": example_index,
                    "reason": "no_gold_tokens_mapped",
                    "edge_count": edge_count,
                    "node_count": node_count,
                    "passage_count": passage_count,
                    **diagnostics,
                }
            )
            continue

        input_rows.append(inputs.cpu().numpy().astype(np.int64, copy=False))
        label_rows.append(labels.cpu().numpy().astype(np.int64, copy=False))
        puzzle_ids.append(int(QUESTION_TYPE_TO_PUZZLE_ID.get(example.evidence.question_type, QUESTION_TYPE_TO_PUZZLE_ID["other"])))
        labeled_token_counts.append(int(diagnostics["labeled_token_count"]))
        row_metadata.append(
            {
                "index": example_index,
                "group_id": example.group_id,
                "question_id": example.evidence.question_id,
                "gold_path_len": int(diagnostics["gold_path_len"]),
                "labeled_token_count": int(diagnostics["labeled_token_count"]),
                "label_mode": diagnostics["label_mode"],
                "missing_gold_nodes": list(diagnostics["missing_gold_nodes"]),
                "missing_gold_edges": list(diagnostics["missing_gold_edges"]),
                "ordered_gold_nodes": list(diagnostics["ordered_gold_nodes"]),
                "labeled_node_ids": list(diagnostics["labeled_node_ids"]),
                "source_positions": list(diagnostics["source_positions"]),
                "labeled_positions": list(diagnostics["labeled_positions"]),
            }
        )

    total_examples = len(input_rows)
    invalid_label_reasons = {"missing_gold_path", "no_gold_tokens_mapped"}
    invalid_label_reject_count = sum(1 for item in skipped_examples if item.get("reason") in invalid_label_reasons)
    invalid_label_reject_rate = invalid_label_reject_count / total_seen if total_seen else 0.0
    graph_stats = {
        "examples_seen": total_seen,
        "examples_saved": total_examples,
        "examples_skipped": len(skipped_examples),
        "invalid_label_reject_count": invalid_label_reject_count,
        "invalid_label_reject_rate": invalid_label_reject_rate,
        "mean_edge_count": _mean(edge_counts),
        "max_edge_count": max(edge_counts, default=0),
        "mean_node_count": _mean(node_counts),
        "max_node_count": max(node_counts, default=0),
        "mean_pubmed_passage_count": _mean(passage_counts),
        "mean_gold_path_length": _mean(gold_path_lengths),
        "mean_labeled_token_count": _mean(labeled_token_counts),
        "max_labeled_token_count": max(labeled_token_counts, default=0),
    }
    arrays = TRMDatasetArrays(
        inputs=np.stack(input_rows, axis=0) if input_rows else np.zeros((0, max_len), dtype=np.int64),
        labels=np.stack(label_rows, axis=0) if label_rows else np.zeros((0, max_len), dtype=np.int64),
        puzzle_identifiers=np.asarray(puzzle_ids, dtype=np.int32),
        puzzle_indices=np.arange(total_examples + 1, dtype=np.int32),
        group_indices=np.arange(total_examples + 1, dtype=np.int32),
        metadata={
            "codebook_version": getattr(embedder, "codebook_version", None),
            "frozen_codebook_path": getattr(embedder, "frozen_codebook_path", None),
            "frozen_codebook_metadata": getattr(embedder, "frozen_codebook_metadata", None),
            "pad_id": pad_id,
            "ignore_label_id": ignore_label_id,
            "blank_identifier_id": 0,
            "vocab_size": vocab_size,
            "seq_len": max_len,
            "num_puzzle_identifiers": DEFAULT_TRM_NUM_PUZZLE_IDENTIFIERS,
            "total_groups": total_examples,
            "mean_puzzle_examples": 1.0 if total_examples else 0.0,
            "total_puzzles": total_examples,
            "sets": ["all"],
            "graph_stats": graph_stats,
            "label_stats": {
                "label_mode": DEFAULT_LABEL_MODE,
                "rows_valid": total_examples,
                "rows_rejected": len(skipped_examples),
                "invalid_label_reject_count": invalid_label_reject_count,
                "invalid_label_reject_rate": invalid_label_reject_rate,
            },
        },
        skipped_examples=skipped_examples,
        row_metadata=row_metadata,
    )
    event_logger.log_event(
        "trm_dataset_arrays_built",
        {
            **graph_stats,
            "seq_len": max_len,
            "vocab_size": vocab_size,
            "backend_used": "medical_trm_dataset_builder",
            "latency_ms": (time.perf_counter() - started_at) * 1000.0,
        },
    )
    return arrays


def save_trm_dataset(arrays: TRMDatasetArrays, output_dir: Path, split: str = "train") -> Path:
    started_at = time.perf_counter()
    event_logger = StructuredLogger("trm_dataset_builder", DEFAULT_LOG_DIR)
    save_dir = KaggleEnv.ensure_writeable(output_dir / split)
    save_dir.mkdir(parents=True, exist_ok=True)

    (save_dir / "dataset.json").write_text(json.dumps(arrays.metadata, indent=2, sort_keys=True), encoding="utf-8")
    np.save(save_dir / "all__inputs.npy", arrays.inputs, allow_pickle=False)
    np.save(save_dir / "all__labels.npy", arrays.labels, allow_pickle=False)
    np.save(save_dir / "all__puzzle_identifiers.npy", arrays.puzzle_identifiers, allow_pickle=False)
    np.save(save_dir / "all__puzzle_indices.npy", arrays.puzzle_indices, allow_pickle=False)
    np.save(save_dir / "all__group_indices.npy", arrays.group_indices, allow_pickle=False)
    (save_dir / "row_metadata.json").write_text(
        json.dumps(arrays.row_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if arrays.skipped_examples:
        (save_dir / "skipped_examples.json").write_text(
            json.dumps(arrays.skipped_examples, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    total_bytes = sum(path.stat().st_size for path in save_dir.glob("*") if path.is_file())
    event_logger.log_event(
        "trm_dataset_saved",
        {
            "save_dir": str(save_dir),
            "split": split,
            "examples_saved": int(arrays.inputs.shape[0]),
            "seq_len": int(arrays.inputs.shape[1]) if arrays.inputs.ndim == 2 else 0,
            "total_bytes": int(total_bytes),
            "metadata": dict(arrays.metadata),
            "backend_used": "medical_trm_dataset_builder",
            "latency_ms": (time.perf_counter() - started_at) * 1000.0,
        },
    )
    return save_dir


def _edge_ids_to_ordered_node_path(
    gold_edge_ids: Sequence[str],
    edge_lookup: Mapping[str, KGEdge],
    missing_edge_ids: list[str],
) -> list[str]:
    node_path: list[str] = []
    ordered_edges: list[KGEdge] = []
    for edge_id in gold_edge_ids:
        normalized_edge_id = str(edge_id).removeprefix("edge:").removeprefix("edge_id:")
        edge = edge_lookup.get(normalized_edge_id)
        if edge is None:
            missing_edge_ids.append(str(edge_id))
            continue
        ordered_edges.append(edge)

    for edge_index, edge in enumerate(ordered_edges):
        if not node_path:
            node_path.extend(_orient_initial_edge(edge, ordered_edges[edge_index + 1] if edge_index + 1 < len(ordered_edges) else None))
            continue
        if node_path[-1] == edge.head:
            _append_if_not_repeated(node_path, edge.tail)
        elif node_path[-1] == edge.tail:
            _append_if_not_repeated(node_path, edge.head)
        else:
            _append_if_not_repeated(node_path, edge.head)
            _append_if_not_repeated(node_path, edge.tail)
    return node_path


def _orient_initial_edge(edge: KGEdge, next_edge: KGEdge | None) -> list[str]:
    if next_edge is None:
        return [edge.head, edge.tail]
    next_nodes = {next_edge.head, next_edge.tail}
    if edge.head in next_nodes and edge.tail not in next_nodes:
        return [edge.tail, edge.head]
    return [edge.head, edge.tail]


def _append_if_not_repeated(node_path: list[str], node_id: str) -> None:
    if not node_path or node_path[-1] != node_id:
        node_path.append(node_id)


def _node_tokens_by_id(inputs: torch.Tensor, node_mapping: Mapping[int, str]) -> dict[str, tuple[int, int]]:
    token_by_node: dict[str, tuple[int, int]] = {}
    for position, node_id in sorted(node_mapping.items()):
        if not _is_gold_node_candidate(node_id):
            continue
        token_by_node.setdefault(node_id, (position, int(inputs[position].item())))
    return token_by_node


def _is_gold_node_candidate(node_id: str) -> bool:
    return bool(node_id and node_id != "__pad__" and not node_id.startswith("PMID:"))


def _coerce_inputs(value: Any, *, max_len: int) -> torch.Tensor:
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.to(dtype=torch.long)
    if tensor.ndim == 2:
        if tensor.shape[0] != 1:
            raise ValueError(f"Expected single-example TRM inputs with shape [1,{max_len}], received {tuple(tensor.shape)}.")
        tensor = tensor[0]
    if tensor.shape != (max_len,):
        raise ValueError(f"Expected TRM input row shape ({max_len},), received {tuple(tensor.shape)}.")
    return tensor


def _normalize_node_mapping(value: Any) -> dict[int, str]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[int, str] = {}
    for key, node_id in value.items():
        try:
            position = int(key)
        except (TypeError, ValueError):
            continue
        normalized[position] = str(node_id)
    return normalized


def _mean(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
