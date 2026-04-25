"""PrimeKG source-predicate → canonical-predicate mapping.

Confidence defaults reflect PrimeKG relation reliability heuristics:
curated drug indications/targets score highest; generic "associated_with"
ground-truth-weak relations land at the floor and get pruned at retrieval.
Polarity captures contraindication / absence relations without overloading
the predicate vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PredicateMapping:
    predicate: str
    confidence: float
    polarity: str = "positive"


PRIMEKG_PREDICATE_MAP: dict[str, PredicateMapping] = {
    # drug ↔ disease
    "indication": PredicateMapping("treats", 0.90),
    "off-label use": PredicateMapping("treats", 0.70),
    "contraindication": PredicateMapping("treats", 0.85, polarity="negative"),
    # drug ↔ drug
    "drug_drug": PredicateMapping("interacts_with", 0.75),
    # drug ↔ effect (side-effect)
    "drug_effect": PredicateMapping("causes_adverse_event", 0.70),
    # drug ↔ protein
    "drug_protein": PredicateMapping("targets", 0.85),
    # disease ↔ protein (gene/protein associated with disease)
    "disease_protein": PredicateMapping("gene_associated_with_disease", 0.75),
    # disease ↔ phenotype
    "disease_phenotype_positive": PredicateMapping("manifested_as", 0.80),
    "disease_phenotype_negative": PredicateMapping("manifested_as", 0.80, polarity="negative"),
    # disease ↔ disease
    "disease_disease": PredicateMapping("associated_with", 0.55),
    # phenotype ↔ phenotype / protein
    "phenotype_phenotype": PredicateMapping("associated_with", 0.50),
    "phenotype_protein": PredicateMapping("associated_with", 0.55),
    # protein ↔ protein
    "protein_protein": PredicateMapping("associated_with", 0.55),
    # pathway
    "pathway_protein": PredicateMapping("participates_in", 0.80),
    "pathway_pathway": PredicateMapping("associated_with", 0.55),
    # biological process / molecular function / cellular component
    "bioprocess_protein": PredicateMapping("participates_in", 0.75),
    "bioprocess_bioprocess": PredicateMapping("associated_with", 0.50),
    "bioprocess_molfunc": PredicateMapping("associated_with", 0.50),
    "molfunc_protein": PredicateMapping("associated_with", 0.60),
    "molfunc_molfunc": PredicateMapping("associated_with", 0.50),
    "cellcomp_protein": PredicateMapping("associated_with", 0.60),
    "cellcomp_cellcomp": PredicateMapping("associated_with", 0.50),
    # anatomy
    "anatomy_anatomy": PredicateMapping("associated_with", 0.50),
    "anatomy_protein_present": PredicateMapping("associated_with", 0.60),
    "anatomy_protein_absent": PredicateMapping("associated_with", 0.60, polarity="negative"),
    # exposure
    "exposure_disease": PredicateMapping("associated_with", 0.55),
    "exposure_protein": PredicateMapping("associated_with", 0.55),
    "exposure_bioprocess": PredicateMapping("associated_with", 0.50),
    "exposure_molfunc": PredicateMapping("associated_with", 0.50),
    "exposure_cellcomp": PredicateMapping("associated_with", 0.50),
    "exposure_exposure": PredicateMapping("associated_with", 0.45),
}


DEFAULT_MAPPING = PredicateMapping("associated_with", 0.40)


class SimplePredicateMapper:
    """Phase-4 MVP predicate mapper.

    Unknown source predicates fall back to ``associated_with`` with low
    confidence so they remain in the graph but get pruned at retrieval time.
    """

    def __init__(self, mapping: dict[str, PredicateMapping] | None = None) -> None:
        self.mapping = dict(mapping) if mapping is not None else dict(PRIMEKG_PREDICATE_MAP)

    def map_source_predicate(
        self,
        source_system: str,
        source_predicate: str,
        subject_type: str | None = None,
        object_type: str | None = None,
    ) -> PredicateMapping:
        _ = source_system, subject_type, object_type  # reserved for domain/range in Phase 5
        return self.mapping.get(source_predicate, DEFAULT_MAPPING)
