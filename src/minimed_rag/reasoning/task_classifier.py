"""Rule-based reasoning task classifier.

Maps a free-text biomedical question to a coarse task type that drives
metapath template selection. Output values are aligned with
``configs/schema/metapath_templates.yaml`` ``task_type:`` strings, plus
the ``general`` fallback that triggers template-free k-hop expansion in
``PathFinder``.

The classifier is intentionally rule-based for Phase 6. ML-based intent
classification is a Phase 7+ refinement.
"""

from __future__ import annotations

import re

TASK_TYPES: tuple[str, ...] = (
    "treatment",
    "adverse_effect",
    "mechanism",
    "gene_disease",
    "general",
)

DEFAULT_TASK_TYPE = "general"

_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(?:adverse|side[\s-]?effects?|toxicity|toxic|"
            r"contraindicat(?:e|ed|ion|ions)|warnings?)\b"
        ),
        "adverse_effect",
    ),
    (re.compile(r"\b(?:what|which)\s+(?:gene|mutation|variant|allele)\b"), "gene_disease"),
    (
        re.compile(
            r"\b(?:gene|mutation|mutated|variant|allele|locus|chromosome)\b.*"
            r"\b(?:in|of|for|caus\w*|associat\w*)\b"
        ),
        "gene_disease",
    ),
    (
        re.compile(r"\b(?:mechanism|pathway|mode\s+of\s+action|moa|target\s+of)\b"),
        "mechanism",
    ),
    (
        re.compile(
            r"\b(?:treat|treats|treated|treating|treatment|therapy|cure|drug\s+for|"
            r"medication\s+for|prescribed\s+for|first[-\s]?line)\b"
        ),
        "treatment",
    ),
)


class TaskClassifier:
    """Map a question to a metapath task type via ordered regex rules."""

    def classify(self, query: str) -> str:
        if not query:
            return DEFAULT_TASK_TYPE
        text = query.casefold()
        for pattern, task_type in _RULES:
            if pattern.search(text):
                return task_type
        return DEFAULT_TASK_TYPE
