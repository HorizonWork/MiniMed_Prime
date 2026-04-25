# Phase 4 — Knowledge Graph MVP Runbook

End-to-end setup for the Phase 4 slice: PrimeKG + UMLS (MRCONSO / MRSTY / MRREL)
loaded into Neo4j with the reified-Assertion schema, plus the 3-tier question
entity linker (scispacy + Postgres synonym fallback + Neo4j resolve) powering
`minimed rag query --use-kg`.

## Prerequisites

- Docker (for the compose stack). `make up` brings up Neo4j + Postgres + MinIO +
  OpenSearch + Milvus. Neo4j heap is set to 4 GB + 2 GB page cache in
  `docker-compose.yml` so bulk MRREL loads don't OOM.
- UMLS license (free from NLM). Place `MRCONSO.RRF`, `MRSTY.RRF`, `MRREL.RRF`
  under `data/umls/local/`.
- ~3 GB free disk for PrimeKG (the Dataverse CSV is ~1.5 GB, extracted).
- ~1 GB free disk for the scispacy UMLS KB on first linker use.

## One-time setup

```bash
# Deps and models
uv sync --extra dev --extra kg
bash scripts/download_scispacy_models.sh

# Infra
make up
bash scripts/bootstrap_postgres_schema.sh

# Data
bash scripts/download_primekg.sh            # writes data/primekg/local/kg.csv
ls data/umls/local/                         # should show MRCONSO.RRF MRSTY.RRF MRREL.RRF
```

## Bootstrap the graph

```bash
make kg-bootstrap
# equivalent to:
#   bash scripts/bootstrap_postgres_schema.sh
#   uv run minimed build-kg apply-schema --graph-version kg_local
#   uv run minimed ingest primekg --release local
#   uv run minimed ingest umls    --release local
#   uv run minimed build-kg crosswalk --graph-version kg_local
```

Ingest time estimates (workstation):

| Stage | Duration | Neo4j rows produced |
| --- | --- | --- |
| `build-kg apply-schema` | <10 s | — |
| `ingest primekg` | 45–90 min | ~130k Entity, ~8M Assertion |
| `ingest umls` (MRCONSO) | ~20 min | ~4M Concept, ~15M Term |
| `ingest umls` (MRSTY) | ~2 min | — (property update) |
| `ingest umls` (MRREL) | 2–3 h | ~25M Assertion (post-filter) |
| `build-kg crosswalk` | ~5 min | Entity.canonical_cui backfill |

## Smoke (faster, non-destructive)

```bash
# PrimeKG only — fast verification the schema/loader work
uv run minimed ingest primekg --release local --limit 100000

# UMLS only — first 500k MRCONSO + 100k MRREL rows
uv run minimed ingest umls --release local --limit-mrconso 500000 --limit-mrrel 100000 --skip mrsty
```

## Ask a question

```bash
uv run minimed rag query "What is metformin used for?" --use-kg
```

Expected output shape:

```
Linked entities:
  - metformin → CUI=C0025598 entity_id=KG:Drug:...  type=CHEMICAL conf=0.94

Graph paths (top 20):
     Metformin --[treats, conf=0.90, src=primekg]--> Type 2 diabetes mellitus
     Metformin --[targets, conf=0.85, src=primekg]--> AMPK
     ...

Average graph confidence: 0.82

Answer:
[no LLM configured — graph_paths populated for inspection]
```

The `--use-kg` path requires Neo4j + Postgres (for the synonym fallback) to
be up. Without `--use-kg`, the pipeline skips linking entirely.

## Sanity checks in Neo4j

```cypher
// Node counts
MATCH (e:Entity)    RETURN count(e) AS entities;
MATCH (c:Concept)   RETURN count(c) AS concepts;
MATCH (a:Assertion) RETURN count(a) AS assertions;
MATCH (e:Entity) WHERE e.canonical_cui IS NOT NULL
RETURN count(e) AS linked_entities;

// Sample PrimeKG treatment
MATCH (drug:Entity {source_system:'primekg'})<-[:SUBJECT]-(a:Assertion {predicate:'treats'})
MATCH (a)-[:OBJECT]->(disease:Entity)
WHERE drug.preferred_name =~ '(?i).*metformin.*'
RETURN drug.preferred_name, a.confidence, disease.preferred_name
ORDER BY a.confidence DESC LIMIT 10;
```

## Troubleshooting

- **`FileNotFoundError: UMLS file not found`** — check `data/umls/local/` has
  `MRCONSO.RRF` / `MRSTY.RRF` / `MRREL.RRF` (not `MRCONSO.RRF.gz`, not nested
  in a release subdir).
- **Neo4j auth failure** — the compose default is `neo4j/minimed`. If you
  changed it, sync `.env` (`NEO4J_PASSWORD=...`).
- **"spacy / scispacy not installed"** — install the `kg` extra and the model:
  `uv sync --extra kg && bash scripts/download_scispacy_models.sh`.
- **Postgres schema missing (concept / term / umls_crosswalk)** — run
  `bash scripts/bootstrap_postgres_schema.sh`. Or use
  `uv run minimed ingest umls --skip-postgres` to load Neo4j only.
- **Neo4j OOM during MRREL** — bump `NEO4J_server_memory_heap_max__size` in
  `docker-compose.yml`, recreate containers (`make down && make up`), and
  rerun. The loader's batch size (5000 by default) already keeps the
  transaction small.
- **Crosswalk links 0 entities** — UMLS MRCONSO uses SAB codes that differ
  from PrimeKG's prefixes. Adjust `_source_id_to_sab_code` in
  `kg_build/umls_crosswalk.py` to cover your releases.
