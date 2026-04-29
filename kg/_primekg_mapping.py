"""PrimeKG ↔ PrimeKG-X schema mapping (single source of truth).

Promoted from scripts/validate_primekg_mapping.py so both the S1 validator
and the S2 builder consume the same tables. The mapping covers:

  * relation: PrimeKG `relation` column → schema edge type name
  * display:  PrimeKG `display_relation` further-disambiguates `drug_protein`
              into target / transporter / enzyme / carrier
  * node:     PrimeKG `x_type`/`y_type` → schema node type name
  * deferred: 6 GO/exposure IS-A relations parked for v0.2 per Final Plan
              §4.2 cap of 40 edges; tracked but never silently dropped
  * helpers:  needs_swap (dynamic edge-direction normalization),
              infer_source / infer_evidence_level (PrimeKG row → evidence
              props the kg.evidence_scoring formula consumes)

The needs_swap helper uses dynamic detection rather than a hardcoded list:
if (x_type, y_type) does not match the schema's (head, tail) but the swap
does, the import physically swaps (x_id ↔ y_id). This self-corrects if
PrimeKG ever fixes the affected relations upstream.
"""

from __future__ import annotations

from typing import Any

# PrimeKG → schema relation map. None = intentionally absent from v0.1
# schema (deferred). __DISPLAY__ = further disambiguation by display_relation.
RELATION_MAP: dict[str, str | None] = {
    "protein_protein": "PROTEIN_PROTEIN",
    "drug_drug": "DRUG_DRUG_INTERACTION",
    "contraindication": "DRUG_CONTRAINDICATION",
    "indication": "DRUG_INDICATION",
    "off-label use": "DRUG_OFF_LABEL",
    "drug_effect": "DRUG_SIDE_EFFECT",
    "drug_protein": "__DISPLAY__",
    "pathway_protein": "GENE_PATHWAY",
    "molfunc_protein": "GENE_MF",
    "bioprocess_protein": "GENE_BP",
    "cellcomp_protein": "GENE_CC",
    "molfunc_molfunc": None,
    "bioprocess_bioprocess": None,
    "cellcomp_cellcomp": None,
    "pathway_pathway": "PATHWAY_PARENT",
    "anatomy_protein_present": "GENE_ANATOMY_EXPRESSED",
    "anatomy_protein_absent": "GENE_ANATOMY_EXPRESSED",
    "anatomy_anatomy": "ANATOMY_PARENT",
    "exposure_disease": "EXPOSURE_DISEASE",
    "exposure_exposure": None,
    "exposure_molfunc": None,
    "exposure_protein": "EXPOSURE_PROTEIN",
    "exposure_bioprocess": "EXPOSURE_BP",
    "exposure_cellcomp": None,
    "disease_protein": "DISEASE_GENE_ASSOC",
    "disease_disease": "DISEASE_PARENT",
    "phenotype_protein": "PHENOTYPE_PROTEIN",
    "phenotype_phenotype": "PHENOTYPE_PARENT",
    "disease_phenotype_negative": "DISEASE_PHENOTYPE_NEG",
    "disease_phenotype_positive": "DISEASE_PHENOTYPE_POS",
}

DISPLAY_DISAMBIGUATION: dict[str, dict[str, str]] = {
    "drug_protein": {
        "target": "DRUG_TARGET",
        "transporter": "DRUG_TRANSPORTER",
        "enzyme": "DRUG_ENZYME",
        "carrier": "DRUG_CARRIER",
    },
}

NODE_TYPE_MAP: dict[str, str] = {
    "gene/protein": "gene_protein",
    "effect/phenotype": "phenotype",
    "disease": "disease",
    "drug": "drug",
    "anatomy": "anatomy",
    "biological_process": "biological_process",
    "molecular_function": "molecular_function",
    "cellular_component": "cellular_component",
    "pathway": "pathway",
    "exposure": "exposure",
}

# 6 PrimeKG relations parked for v0.2 (per docs/v0_2_backlog.md §4.2 cap).
# Tracked at build time; reports/s2_deferred_edges.md enumerates per-relation
# count + 10 sample rows + ≤5%-of-total sanity check.
DEFERRED_RELATIONS: frozenset[str] = frozenset(
    rel for rel, mapped in RELATION_MAP.items() if mapped is None
)

# PrimeKG stores these same-type hierarchy edges as parent → child but the
# v0.1 schema's NL templates ("{phenotype} is a kind of {parent_phenotype}",
# etc., per docs/path_representation_spec.md §4) require child → parent.
# needs_swap() can't detect this because head_type == tail_type — we override
# with an explicit list. Verified empirically against 8 random rows per
# relation. anatomy_anatomy is intentionally absent: PrimeKG already stores
# it child → part-of-parent, matching the schema template "{anatomy} is part
# of {parent_anatomy}".
SAME_TYPE_HIERARCHY_SWAP: frozenset[str] = frozenset({
    "disease_disease",
    "phenotype_phenotype",
    "pathway_pathway",
})


def hierarchy_needs_swap(row: dict[str, Any]) -> bool:
    """True iff this is a same-type *_PARENT row PrimeKG stores in the wrong
    direction (parent → child) and we must canonicalize to child → parent."""
    return row.get("relation", "") in SAME_TYPE_HIERARCHY_SWAP


def map_relation(row: dict[str, Any]) -> str | None:
    """Resolve a PrimeKG row to a schema edge type name.

    Returns None when the relation is in DEFERRED_RELATIONS or is an
    unknown/missing mapping. The caller distinguishes the two via membership
    in DEFERRED_RELATIONS.
    """
    rel = row.get("relation", "")
    mapped = RELATION_MAP.get(rel)
    if mapped == "__DISPLAY__":
        display = (row.get("display_relation") or "").strip()
        return DISPLAY_DISAMBIGUATION.get(rel, {}).get(display)
    return mapped


def map_node_type(primekg_type: str) -> str | None:
    """Resolve a PrimeKG node-type string to a schema node type name."""
    return NODE_TYPE_MAP.get(primekg_type)


def needs_swap(row: dict[str, Any], schema_edges: dict[str, Any]) -> bool:
    """Decide whether to swap (x, y) so the edge matches schema direction.

    Returns True iff (x_type, y_type) does not match (head, tail) declared
    in the schema for this edge, but (y_type, x_type) does. Short-circuits
    to False for self-relations (head_type == tail_type — e.g.
    PROTEIN_PROTEIN) and for symmetric edges, where direction is irrelevant.
    """
    edge_name = map_relation(row)
    if edge_name is None or edge_name not in schema_edges:
        return False
    decl = schema_edges[edge_name]
    if decl.head_type == decl.tail_type:
        return False
    if decl.symmetric:
        return False
    x_type = map_node_type(row.get("x_type", ""))
    y_type = map_node_type(row.get("y_type", ""))
    if x_type is None or y_type is None:
        return False
    if decl.head_type == x_type and decl.tail_type == y_type:
        return False
    return decl.head_type == y_type and decl.tail_type == x_type


def infer_source(row: dict[str, Any]) -> str:
    """Infer evidence `source` from a PrimeKG row.

    Mirrors the heuristic in scripts/validate_primekg_mapping.py used to
    annotate the calibration sample, so production weights line up with the
    distribution coefficients were calibrated against.
    """
    rel = row.get("relation", "")
    x_src = row.get("x_source", "")
    y_src = row.get("y_source", "")
    if x_src == "DrugBank" or y_src == "DrugBank":
        return "DrugBank"
    if "disease" in rel:
        return "DisGeNET"
    if rel.startswith(("molfunc", "bioprocess", "cellcomp")):
        return "GO"
    if rel.startswith("pathway"):
        return "Reactome"
    if rel.startswith("anatomy"):
        return "UBERON"
    if "phenotype" in rel:
        return "HPO"
    return "MONDO"


def infer_evidence_level(row: dict[str, Any]) -> str:
    """Infer evidence_level bucket from a PrimeKG row.

    Same heuristic as the calibration sample: indication/contraindication →
    `approved`; disease-* relations → DisGeNET `score>0.5`; everything else
    falls back to `manual_review`.
    """
    rel = row.get("relation", "")
    if rel in ("indication", "contraindication", "off-label use"):
        return "approved"
    if "disease" in rel:
        return "score>0.5"
    return "manual_review"
