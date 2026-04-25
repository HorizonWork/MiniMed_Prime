"""UMLS MRREL → canonical predicate mapping.

UMLS represents relation semantics in two fields: ``REL`` (a broad class) and
``RELA`` (a specific attribute). We map the combination to our canonical
predicate vocabulary (see ``configs/schema/predicates.yaml``).

Confidence defaults reflect the reliability of the relation class:
- Hierarchical (``PAR``/``CHD``, ``RN``/``RB``): high confidence, generic
  ``associated_with`` predicate since ``is_a`` is not in the canonical set yet.
- Named clinical attributes (``may_treat``, ``causative_agent_of``, …): map to
  specific canonical predicates with benchmark-grade confidence.
- Unnamed ``RO`` + empty ``RELA``: fall back to low-confidence
  ``associated_with``.
"""

from __future__ import annotations

from minimed_rag.ingestion.primekg.predicate_map import PredicateMapping

# RELA → canonical mapping. These override the REL-class defaults below.
RELA_MAP: dict[str, PredicateMapping] = {
    "may_treat": PredicateMapping("treats", 0.75),
    "may_prevent": PredicateMapping("treats", 0.60),
    "treats": PredicateMapping("treats", 0.85),
    "prevents": PredicateMapping("treats", 0.75),
    "contraindicated_with_disease": PredicateMapping("treats", 0.80, polarity="negative"),
    "has_contraindication": PredicateMapping("treats", 0.80, polarity="negative"),
    "has_causative_agent": PredicateMapping("causes_adverse_event", 0.70),
    "causative_agent_of": PredicateMapping("causes_adverse_event", 0.70),
    "mechanism_of_action_of": PredicateMapping("targets", 0.75),
    "has_mechanism_of_action": PredicateMapping("targets", 0.75),
    "has_target": PredicateMapping("targets", 0.80),
    "target_of": PredicateMapping("targets", 0.80),
    "gene_associated_with_disease": PredicateMapping("gene_associated_with_disease", 0.80),
    "disease_has_associated_gene": PredicateMapping("gene_associated_with_disease", 0.80),
    "has_manifestation": PredicateMapping("manifested_as", 0.75),
    "manifestation_of": PredicateMapping("manifested_as", 0.75),
    "has_symptom": PredicateMapping("manifested_as", 0.70),
    "symptom_of": PredicateMapping("manifested_as", 0.70),
    "part_of_pathway": PredicateMapping("participates_in", 0.75),
    "pathway_has_gene_product": PredicateMapping("participates_in", 0.75),
}

# REL class → default mapping when RELA is empty / unmapped.
REL_MAP: dict[str, PredicateMapping] = {
    "PAR": PredicateMapping("associated_with", 0.80),  # is-a parent
    "CHD": PredicateMapping("associated_with", 0.80),  # is-a child
    "RN": PredicateMapping("associated_with", 0.70),  # narrower
    "RB": PredicateMapping("associated_with", 0.70),  # broader
    "RO": PredicateMapping("associated_with", 0.50),  # related other
}

DEFAULT_MAPPING = PredicateMapping("associated_with", 0.40)


class UMLSPredicateMapper:
    def __init__(
        self,
        rela_map: dict[str, PredicateMapping] | None = None,
        rel_map: dict[str, PredicateMapping] | None = None,
    ) -> None:
        self.rela_map = dict(rela_map) if rela_map is not None else dict(RELA_MAP)
        self.rel_map = dict(rel_map) if rel_map is not None else dict(REL_MAP)

    def map(self, rel: str, rela: str | None) -> PredicateMapping:
        if rela:
            mapped = self.rela_map.get(rela)
            if mapped is not None:
                return mapped
        return self.rel_map.get(rel, DEFAULT_MAPPING)
