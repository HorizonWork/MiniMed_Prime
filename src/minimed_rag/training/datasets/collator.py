"""Pseudocode TRM collator."""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence


def trm_collator(batch: list[dict]) -> dict:
    return {
        "question_tokens": pad_sequence(
            [item["question_tokens"] for item in batch], batch_first=True
        ),
        "option_tokens": pad_sequence([item["option_tokens"] for item in batch], batch_first=True),
        "entity_path": pad_sequence([item["entity_path"] for item in batch], batch_first=True),
        "relation_path": pad_sequence([item["relation_path"] for item in batch], batch_first=True),
        "edge_confidences": pad_sequence(
            [item["edge_confidences"] for item in batch], batch_first=True
        ).float(),
        "task_type_id": torch.stack([item["task_type_id"] for item in batch]),
        "metapath_id": torch.stack([item["metapath_id"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
    }
