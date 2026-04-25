"""UMLS → PrimeKG canonical-CUI crosswalk.

Walks the Postgres ``umls_crosswalk`` table ((sab, code, cui) triples loaded
from MRCONSO) and backfills ``Entity.canonical_cui`` in Neo4j by matching
PrimeKG source_ids to (SAB, CODE) pairs. A dedicated ``:HAS_CONCEPT`` edge
is also MERGEd so paths from :Entity to :Concept are one hop.

The set of PrimeKG ID patterns → UMLS (SAB, CODE) mappings is approximate;
it covers the largest PrimeKG sources (MONDO, DrugBank, HPO, UBERON, GO,
ChEBI, NCBI Gene) and is conservative elsewhere.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# PrimeKG entity_type + source_id → (SAB, CODE) candidates.
# Emitted in preference order; first Postgres match wins.
def _source_id_to_sab_code(entity_type: str, source_id: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    if source_id.startswith("MONDO:"):
        candidates.append(("MONDO", source_id))
    elif source_id.startswith("DOID:"):
        candidates.append(("DOID", source_id))
    elif source_id.startswith("UBERON:"):
        candidates.append(("UBERON", source_id))
    elif source_id.startswith("HP:"):
        candidates.append(("HPO", source_id))
    elif source_id.startswith("GO:"):
        candidates.append(("GO", source_id))
    elif source_id.startswith("CHEBI:"):
        candidates.append(("CHEBI", source_id))
    elif source_id.startswith(("R-HSA-", "R-MMU-", "REACT:")):
        # PrimeKG pathway nodes use Reactome IDs (R-HSA-xxxxx for human,
        # R-MMU- for mouse). UMLS SAB is "REACTOME"; some local releases
        # also load it as "REACT" — try both.
        candidates.append(("REACTOME", source_id))
        candidates.append(("REACT", source_id))
    elif source_id.startswith(("MESH:", "MSH:")):
        # MeSH descriptors used by exposures + some PrimeKG disease/chemical
        # entries imported from CTD. UMLS SAB is "MSH"; the colon is part of
        # the local code in some loads, so try both.
        bare = source_id.split(":", 1)[1] if ":" in source_id else source_id
        candidates.append(("MSH", source_id))
        candidates.append(("MSH", bare))
    elif source_id.startswith("CTD:"):
        # CTD ids are typically MeSH-based; try MSH after CTD.
        bare = source_id.split(":", 1)[1]
        candidates.append(("CTD", source_id))
        candidates.append(("MSH", bare))
    elif source_id.startswith("DB") and entity_type in ("Drug", "Chemical"):
        candidates.append(("DRUGBANK", source_id))
        candidates.append(("RXNORM", source_id))
    elif source_id.startswith(("ENSG", "ENST", "ENSP")) and entity_type in (
        "Gene", "GeneProtein", "Protein"
    ):
        # Ensembl gene/transcript/protein. UMLS sometimes loads Ensembl as
        # "OMIM" or "HGNC" cross-refs; we fall back to those.
        candidates.append(("ENSEMBL", source_id))
        candidates.append(("HGNC", source_id))
    elif source_id.isdigit() and entity_type in ("Gene", "GeneProtein", "Protein"):
        # PrimeKG gene IDs are NCBI gene IDs as bare integers.
        candidates.append(("NCBI", source_id))
    return candidates


def _prefix_of(source_id: str) -> str:
    """Return a short label used to bucket per-prefix match counts."""
    if not source_id:
        return "(empty)"
    for token in (
        "MONDO:", "DOID:", "UBERON:", "HP:", "GO:", "CHEBI:",
        "R-HSA-", "R-MMU-", "REACT:",
        "MESH:", "MSH:", "CTD:",
    ):
        if source_id.startswith(token):
            return token.rstrip(":-")
    for token in ("ENSG", "ENST", "ENSP"):
        if source_id.startswith(token):
            return token
    if source_id.startswith("DB"):
        return "DB"
    if source_id.isdigit():
        return "(numeric)"
    return "(other)"


@dataclass
class CrosswalkRunner:
    neo4j: Any
    postgres: Any
    batch_size: int = 1000
    counts: dict[str, int] = field(
        default_factory=lambda: {"examined": 0, "matched": 0, "linked": 0}
    )
    per_prefix: dict[str, dict[str, int]] = field(default_factory=dict)

    def _bump_prefix(self, prefix: str, key: str) -> None:
        bucket = self.per_prefix.setdefault(prefix, {"examined": 0, "matched": 0})
        bucket[key] += 1

    def prefix_match_rate_summary(self) -> str:
        """Render the per-prefix match table as multiline text for logging."""
        if not self.per_prefix:
            return "(no entities examined)"
        rows = sorted(
            self.per_prefix.items(),
            key=lambda kv: kv[1]["examined"],
            reverse=True,
        )
        lines = [f"  {'prefix':<14} {'examined':>10} {'matched':>10} {'rate':>8}"]
        for prefix, bucket in rows:
            ex = bucket["examined"]
            mt = bucket["matched"]
            rate = (mt / ex) if ex else 0.0
            lines.append(f"  {prefix:<14} {ex:>10d} {mt:>10d} {rate:>7.1%}")
        return "\n".join(lines)

    def iter_primekg_entities(self, graph_version: str | None = None) -> Iterator[dict]:
        cypher = (
            "MATCH (e:Entity) WHERE e.source_system = 'primekg' "
            + ("AND e.graph_version = $gv " if graph_version else "")
            + "AND (e.canonical_cui IS NULL OR e.canonical_cui = '') "
            "RETURN e.entity_id AS entity_id, e.entity_type AS entity_type, "
            "e.source_id AS source_id"
        )
        params = {"gv": graph_version} if graph_version else {}
        yield from self.neo4j.query(cypher, params)

    def resolve_cui(self, entity_type: str, source_id: str) -> tuple[str, str, str] | None:
        for sab, code in _source_id_to_sab_code(entity_type, source_id):
            rows = self.postgres.execute(
                "SELECT cui FROM umls_crosswalk WHERE sab = %(sab)s AND code = %(code)s LIMIT 1",
                {"sab": sab, "code": code},
            )
            if rows:
                return sab, code, rows[0][0]
        return None

    def run(self, graph_version: str | None = None) -> dict[str, int]:
        batch: list[dict] = []
        for entity in self.iter_primekg_entities(graph_version):
            self.counts["examined"] += 1
            prefix = _prefix_of(entity.get("source_id", "") or "")
            self._bump_prefix(prefix, "examined")
            resolved = self.resolve_cui(entity["entity_type"], entity["source_id"])
            if resolved is None:
                continue
            sab, code, cui = resolved
            batch.append(
                {
                    "entity_id": entity["entity_id"],
                    "cui": cui,
                    "match": f"{sab}:{code}",
                }
            )
            self.counts["matched"] += 1
            self._bump_prefix(prefix, "matched")
            if len(batch) >= self.batch_size:
                self._flush(batch)
                batch = []
        if batch:
            self._flush(batch)
        logger.info("umls_crosswalk: %s", self.counts)
        logger.info("umls_crosswalk per-prefix:\n%s", self.prefix_match_rate_summary())
        return dict(self.counts)

    def _flush(self, rows: Iterable[dict]) -> None:
        cypher = """
        UNWIND $rows AS row
        MATCH (e:Entity {entity_id: row.entity_id})
        MATCH (c:Concept {cui: row.cui})
        MERGE (e)-[r:HAS_CONCEPT]->(c)
        SET r.match = row.match,
            e.canonical_cui = row.cui
        """
        rows_list = list(rows)
        self.neo4j.run(cypher, rows=rows_list)
        self.counts["linked"] += len(rows_list)
