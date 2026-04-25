"""Unit tests for the Phase 6 TaskClassifier."""

from __future__ import annotations

import pytest

from minimed_rag.reasoning.task_classifier import DEFAULT_TASK_TYPE, TASK_TYPES, TaskClassifier


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What gene is mutated in cystic fibrosis?", "gene_disease"),
        ("Which gene causes sickle cell anemia?", "gene_disease"),
        ("What mutation in BRCA1 is associated with breast cancer?", "gene_disease"),
        ("What is the mechanism of metformin?", "mechanism"),
        ("Describe the pathway involved in beta-oxidation.", "mechanism"),
        ("What is the mode of action of aspirin?", "mechanism"),
        ("What treats type 2 diabetes?", "treatment"),
        ("First-line therapy for hypertension?", "treatment"),
        ("Drug for migraine prophylaxis?", "treatment"),
        ("What are the side effects of aspirin?", "adverse_effect"),
        ("Adverse events of NSAIDs?", "adverse_effect"),
        ("Is warfarin contraindicated in pregnancy?", "adverse_effect"),
        ("Tell me about the cell.", "general"),
        ("", "general"),
    ],
)
def test_classify_known_intents(question: str, expected: str) -> None:
    assert TaskClassifier().classify(question) == expected


def test_default_falls_back_to_general() -> None:
    assert TaskClassifier().classify("xyzzy plugh") == DEFAULT_TASK_TYPE


def test_classify_is_case_insensitive() -> None:
    assert TaskClassifier().classify("WHAT GENE Mutated In CF?") == "gene_disease"


def test_task_types_contains_all_intents() -> None:
    for t in ("treatment", "adverse_effect", "mechanism", "gene_disease", "general"):
        assert t in TASK_TYPES
