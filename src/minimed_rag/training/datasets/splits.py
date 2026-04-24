"""Pseudocode dataset splitting."""

from __future__ import annotations


def assign_split(task_id: str, train_ratio: float = 0.8, val_ratio: float = 0.1) -> str:
    bucket = hash(task_id) % 1000 / 1000
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"
