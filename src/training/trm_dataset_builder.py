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
    """Map gold KG edge path to TRM labels using Layer 2 token positions.

    Current Layer 2 exposes token-position provenance as node IDs, not edge IDs. This
    function therefore labels the token positions of the ordered endpoint nodes touched
    by the gold edges and ignores all other positions.
    """
    inputs = _coerce_inputs(embedder_output["inputs"], max_len=max_len)
    node_mapping = _normalize_node_mapping(embedder_output.get("node_mapping", {}))
    position_by_node = {
        node_id: position
        for position, node_id in node_mapping.items()
        if isinstance(node_id, str) and node_id and node_id != "__pad__" and not node_id.startswith("PMID:")
    }
    edge_lookup = {edge.edge_id: edge for edge in evidence.subgraph_edges}

    labels = torch.full((max_len,), int(ignore_label_id), dtype=torch.long)
    missing_edge_ids: list[str] = []
    missing_node_ids: list[str] = []
    labeled_positions: list[int] = []

    for node_id in _edge_ids_to_node_path(gold_edge_ids, edge_lookup, missing_edge_ids):
        position = position_by_node.get(node_id)
        if position is None:
            if node_id not in missing_node_ids:
                missing_node_ids.append(node_id)
            continue
        labels[position] = inputs[position]
        if position not in labeled_positions:
            labeled_positions.append(position)

    diagnostics = {
        "gold_edge_ids": list(gold_edge_ids),
        "missing_edge_ids": missing_edge_ids,
        "missing_node_ids": missing_node_ids,
        "labeled_positions": sorted(labeled_positions),
        "labeled_token_count": len(labeled_positions),
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
                    "reason": "missing_gold_edge_ids",
                    "edge_count": edge_count,
                    "node_count": node_count,
                    "passage_count": passage_count,
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

    total_examples = len(input_rows)
    graph_stats = {
        "examples_seen": total_seen,
        "examples_saved": total_examples,
        "examples_skipped": len(skipped_examples),
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
        },
        skipped_examples=skipped_examples,
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


def _edge_ids_to_node_path(
    gold_edge_ids: Sequence[str],
    edge_lookup: Mapping[str, KGEdge],
    missing_edge_ids: list[str],
) -> list[str]:
    node_path: list[str] = []
    for edge_id in gold_edge_ids:
        edge = edge_lookup.get(str(edge_id).removeprefix("edge:").removeprefix("edge_id:"))
        if edge is None:
            missing_edge_ids.append(str(edge_id))
            continue
        if not node_path:
            node_path.extend([edge.head, edge.tail])
            continue
        if node_path[-1] == edge.head:
            node_path.append(edge.tail)
        elif node_path[-1] == edge.tail:
            node_path.append(edge.head)
        else:
            node_path.extend([edge.head, edge.tail])
    deduped_path: list[str] = []
    for node_id in node_path:
        if node_id not in deduped_path:
            deduped_path.append(node_id)
    return deduped_path


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
