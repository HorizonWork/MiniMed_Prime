"""Pseudocode MedReason normalizer."""

from __future__ import annotations

from minimed_rag.ingestion.base import NormalizedRecord, RawRecord


def normalize_medreason_task(record: RawRecord) -> list[NormalizedRecord]:
    payload = record.payload
    return [
        NormalizedRecord(
            "reasoning_task",
            {
                "task_id": payload["id"],
                "question": payload["question"],
                "options": payload.get("options", []),
                "correct_answer": payload.get("answer"),
                "task_type": payload.get("task_type", "unknown"),
            },
        )
    ]
