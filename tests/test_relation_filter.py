from __future__ import annotations

from src.retrieval import (
    canonicalize_question_type,
    infer_question_type,
    resolve_relation_filter,
    summarize_relation_filter_stats,
)


def test_relation_alias_normalization_drugdrug_matches_drug_drug() -> None:
    resolution = resolve_relation_filter(
        question_type="drug_interaction",
        available_relations={"drug_drug", "contraindication"},
        explicit_override={"drugdrug"},
    )

    assert resolution.matched_relations == {"drug_drug"}
    assert resolution.requested_but_missing_relations == set()


def test_relation_alias_normalization_off_label_use_matches_off_label_space() -> None:
    resolution = resolve_relation_filter(
        question_type="drug_interaction",
        available_relations={"off-label use"},
        explicit_override={"off_label_use"},
    )

    assert resolution.matched_relations == {"off-label use"}
    assert resolution.requested_but_missing_relations == set()


def test_relation_family_expansion_disease_symptom() -> None:
    resolution = resolve_relation_filter(
        question_type="diagnosis",
        available_relations={"disease_phenotype_positive", "disease_phenotype_negative"},
        explicit_override={"disease_symptom"},
    )

    assert resolution.matched_relations == {"disease_phenotype_positive", "disease_phenotype_negative"}


def test_relation_family_expansion_gene_disease_maps_to_disease_protein() -> None:
    resolution = resolve_relation_filter(
        question_type="etiology",
        available_relations={"disease_protein"},
        explicit_override={"gene_disease"},
    )

    assert resolution.matched_relations == {"disease_protein"}


def test_requested_but_missing_relations_are_reported() -> None:
    resolution = resolve_relation_filter(
        question_type="drug_interaction",
        available_relations={"drug_drug"},
        explicit_override={"contraindication"},
    )

    assert resolution.matched_relations == set()
    assert resolution.requested_but_missing_relations == {"contraindication"}


def test_factoid_canonicalizes_to_other_for_routing() -> None:
    assert canonicalize_question_type("factoid") == "other"
    assert canonicalize_question_type("factoid", for_routing=False) == "factoid"


def test_infer_question_type_supports_puzzle_identifiers_hint() -> None:
    inferred = infer_question_type(record={"puzzle_identifiers": [2]}, question="irrelevant", fallback_label="other")
    assert inferred == "dosage"


def test_infer_question_type_detects_anatomy_factoid_questions() -> None:
    inferred = infer_question_type(
        question="The urogenital diaphragm is composed of all of the following except:",
        fallback_label="other",
    )

    assert inferred == "factoid"


def test_factoid_relation_resolution_keeps_anatomy_edges() -> None:
    resolution = resolve_relation_filter(
        question_type="factoid",
        available_relations={"anatomy_anatomy", "drug_drug"},
    )

    assert "anatomy_anatomy" in resolution.matched_relations


def test_summarize_relation_filter_stats_contains_observability_fields() -> None:
    payload = summarize_relation_filter_stats(
        question_type="drug_interaction",
        requested_relations={"drug_drug"},
        matched_relations={"drug_drug"},
        edges_before=12,
        edges_after=5,
        node_count_after=6,
        fallback_stage="none",
        override_source="question_type_default",
        use_relation_filter=True,
    )

    assert payload["question_type_used"] == "drug_interaction"
    assert payload["fallback_stage_used"] == "none"
    assert payload["edges_before"] == 12
    assert payload["edges_after"] == 5
    assert payload["node_count_after"] == 6
