# MiniMed RAG — Biomedical KG-RAG + TRM Reasoning

Production scaffold for biomedical KG construction, hybrid KG-RAG retrieval, reasoning path generation, and TRM-style path training.

See [Research Roadmap](docs/research_roadmap.md) for the benchmark-first agile plan.

---

## Quick start (local)

### 1. Install dependencies

```bash
make install          # uv sync --extra dev
```

### 2. Start local services

```bash
make up               # docker compose up -d (postgres, neo4j, minio, opensearch, milvus)
```

Wait ~10 s for services to be ready, then verify:

```bash
docker compose ps     # all services should show "running"
```

### 3. Configure environment

```bash
cp .env.example .env  # edit credentials if needed (defaults work with make up)
```

### 4. Run the API

```bash
make api              # uvicorn on http://localhost:8000
```

Health check:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

Interactive docs: <http://localhost:8000/docs>

### 5. Use the CLI

```bash
uv run minimed --help

# Ingest a source
uv run minimed ingest source primekg --release local

# Build the canonical KG
uv run minimed build-kg --graph-version kg_local

# Build serving indexes
uv run minimed build-indexes all --graph-version kg_local

# Ask a RAG question
uv run minimed rag ask "What drugs treat type 2 diabetes?" --graph-version kg_local

# Generate reasoning paths
uv run minimed generate-paths --graph-version kg_local

# Train TRM
uv run minimed train-trm --config configs/training/trm_training.yaml --graph-version kg_local

# Evaluate
uv run minimed evaluate all --graph-version kg_local
```

---

## Development

```bash
make test             # run unit + smoke tests
make smoke            # fast fail-first smoke pass
make lint             # ruff check + format check
make lint-fix         # auto-fix all lint issues
make type-check       # mypy
```

### Stop services

```bash
make down
```

---

## Project layout

```text
src/minimed_rag/
├── api/          # FastAPI app + routes
├── cli/          # Typer CLI entrypoints
├── common/       # Config, logging, ids, hashing
├── domain/       # Pydantic domain models
├── ingestion/    # Source-specific ingestion pipelines
├── nlp/          # NER, entity linking, relation extraction, chunking
├── kg_build/     # Canonical KG construction
├── normalization/# Entity resolution, predicate mapping
├── indexing/     # Embedding, vector, graph, search indexes
├── retrieval/    # Hybrid retrieval + RAG pipeline
├── reasoning/    # Metapath planning, path generation
├── training/     # TRM model, datasets, trainer
├── evaluation/   # Eval harnesses
├── schema_registry/ # Ontology configs loaded from configs/schema/
└── storage/      # DB/store clients + repositories

configs/          # YAML configs (env, pipeline, schema, training)
data_contracts/   # JSON Schema + SHACL + SQL schemas
pipelines/        # Airflow / Prefect DAGs
tests/
├── unit/         # Fast, no external deps
├── integration/  # Require live services
├── e2e/          # End-to-end
└── contract/     # Data contract validation
```

---

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_ENV` | `local` | Config profile (`local`, `dev`, `staging`, `prod`) |
| `GRAPH_VERSION` | `kg_local` | KG version tag used across pipelines |
| `POSTGRES_DSN` | `postgresql://minimed:minimed@localhost:5432/minimed` | Metadata DB |
| `NEO4J_URI` | `bolt://localhost:7687` | Graph DB |
| `NEO4J_USER` | `neo4j` | — |
| `NEO4J_PASSWORD` | `minimed` | — |
| `MINIO_ENDPOINT` | `http://localhost:9000` | Object store |
| `MINIO_ACCESS_KEY` | `minio` | — |
| `MINIO_SECRET_KEY` | `minio123` | — |
| `MILVUS_URI` | `http://localhost:19530` | Vector DB |
| `OPENSEARCH_URL` | `http://localhost:9200` | BM25 search |
