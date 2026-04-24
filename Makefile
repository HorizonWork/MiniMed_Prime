.PHONY: install test lint type-check smoke up down api \
        quality ingest-primekg ingest-umls build-kg build-indexes train-trm evaluate

# ── Dev setup ────────────────────────────────────────────────────────────────

install:
	uv sync --extra dev

# ── Quality ───────────────────────────────────────────────────────────────────

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

lint-fix:
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

type-check:
	uv run mypy src/minimed_rag/

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	uv run pytest tests/unit/ -v

smoke:
	uv run pytest tests/unit/ -v -x --tb=short

# ── Services (Docker) ─────────────────────────────────────────────────────────

up:
	docker compose up -d

down:
	docker compose down

# ── Local API server ──────────────────────────────────────────────────────────

api:
	uv run uvicorn minimed_rag.api.main:app --host 0.0.0.0 --port 8000 --reload

# ── Pipeline targets ──────────────────────────────────────────────────────────

quality:
	minimed evaluate all --graph-version $${GRAPH_VERSION:-kg_local}

ingest-primekg:
	minimed ingest primekg --release $${RELEASE:-local}

ingest-umls:
	minimed ingest umls --release $${RELEASE:-local}

build-kg:
	minimed build-kg --graph-version $${GRAPH_VERSION:-kg_local}

build-indexes:
	minimed build-indexes all --graph-version $${GRAPH_VERSION:-kg_local}

train-trm:
	minimed train-trm --config configs/training/trm_training.yaml --graph-version $${GRAPH_VERSION:-kg_local}

evaluate:
	minimed evaluate all --graph-version $${GRAPH_VERSION:-kg_local}
