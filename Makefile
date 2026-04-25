.PHONY: install test lint type-check smoke up down api \
        quality ingest-primekg ingest-umls build-kg build-indexes train-trm evaluate \
        kg-bootstrap kg-schema kg-crosswalk kg-full

UV_CACHE_DIR ?= /tmp/uv-cache
export UV_CACHE_DIR
UV := uv

# ── Dev setup ────────────────────────────────────────────────────────────────

install:
	$(UV) sync --extra dev

# ── Quality ───────────────────────────────────────────────────────────────────

lint:
	$(UV) run ruff check src/ tests/
	$(UV) run ruff format --check src/ tests/

lint-fix:
	$(UV) run ruff check --fix src/ tests/
	$(UV) run ruff format src/ tests/

type-check:
	$(UV) run mypy src/minimed_rag/

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	$(UV) run pytest tests/unit/ -v

smoke:
	$(UV) run pytest tests/unit/ -v -x --tb=short

# ── Services (Docker) ─────────────────────────────────────────────────────────

up:
	docker compose up -d

down:
	docker compose down

# ── Local API server ──────────────────────────────────────────────────────────

api:
	$(UV) run uvicorn minimed_rag.api.main:app --host 0.0.0.0 --port 8000 --reload

# ── Pipeline targets ──────────────────────────────────────────────────────────

quality:
	minimed evaluate all --graph-version $${GRAPH_VERSION:-kg_local}

ingest-primekg:
	$(UV) run minimed ingest primekg --release $${RELEASE:-local}

ingest-umls:
	$(UV) run minimed ingest umls --release $${RELEASE:-local}

# ── Phase 4 KG convenience targets ───────────────────────────────────────────

kg-schema:
	bash scripts/bootstrap_postgres_schema.sh
	$(UV) run minimed build-kg apply-schema --graph-version $${GRAPH_VERSION:-kg_local}

kg-crosswalk:
	$(UV) run minimed build-kg crosswalk --graph-version $${GRAPH_VERSION:-kg_local}

kg-bootstrap: kg-schema ingest-primekg ingest-umls kg-crosswalk
	@echo "Phase 4 KG bootstrap complete."

kg-full: kg-bootstrap
	@echo "End-to-end KG build finished (primekg + umls + crosswalk)."

build-kg:
	minimed build-kg --graph-version $${GRAPH_VERSION:-kg_local}

build-indexes:
	minimed build-indexes all --graph-version $${GRAPH_VERSION:-kg_local}

train-trm:
	minimed train-trm --config configs/training/trm_training.yaml --graph-version $${GRAPH_VERSION:-kg_local}

evaluate:
	minimed evaluate all --graph-version $${GRAPH_VERSION:-kg_local}
