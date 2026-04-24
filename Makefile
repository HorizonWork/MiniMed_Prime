.PHONY: up down api quality ingest-primekg ingest-umls build-kg build-indexes train-trm evaluate
up:
	docker compose up -d
down:
	docker compose down
api:
	uvicorn biomed_kg.api.main:app --host 0.0.0.0 --port 8000 --reload
quality:
	biomedkg evaluate all --graph-version $${GRAPH_VERSION:-kg_local}
ingest-primekg:
	biomedkg ingest primekg --release $${RELEASE:-local}
ingest-umls:
	biomedkg ingest umls --release $${RELEASE:-local}
build-kg:
	biomedkg build-kg --graph-version $${GRAPH_VERSION:-kg_local}
build-indexes:
	biomedkg build-indexes all --graph-version $${GRAPH_VERSION:-kg_local}
train-trm:
	biomedkg train-trm --config configs/training/trm_training.yaml --graph-version $${GRAPH_VERSION:-kg_local}
evaluate:
	biomedkg evaluate all --graph-version $${GRAPH_VERSION:-kg_local}
