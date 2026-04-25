"""FastAPI / CLI shared dependency factories.

Lazily wires Phase 4 components so import-time cost stays low for tools
that only need a subset (e.g. `minimed ingest primekg` doesn't need
scispacy, but `minimed rag query --use-kg` does).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from minimed_rag.common.config import get_settings
from minimed_rag.kg.graph_client import GraphClient
from minimed_rag.storage.neo4j import Neo4j
from minimed_rag.storage.neo4j import connect as neo4j_connect
from minimed_rag.storage.postgres import Postgres
from minimed_rag.storage.postgres import connect as postgres_connect
from minimed_rag.storage.repositories.concept_repo import ConceptRepository
from minimed_rag.storage.repositories.term_repo import TermRepository


class ServiceContainer:
    """Minimal DI container. Kept for backwards-compat with existing imports."""

    def __init__(self) -> None:
        self.services: dict[str, Any] = {}

    def get(self, name: str) -> Any:
        return self.services[name]

    def set(self, name: str, value: Any) -> None:
        self.services[name] = value


container = ServiceContainer()


def get_services() -> ServiceContainer:
    return container


# ── Cached factories ─────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_neo4j() -> Neo4j:
    settings = get_settings()
    return neo4j_connect(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        settings.neo4j_database,
    )


@lru_cache(maxsize=1)
def get_postgres() -> Postgres:
    settings = get_settings()
    return postgres_connect(settings.postgres_dsn)


@lru_cache(maxsize=1)
def get_graph_client() -> GraphClient:
    return GraphClient(neo4j=get_neo4j())


@lru_cache(maxsize=1)
def get_term_repository() -> TermRepository:
    return TermRepository(postgres=get_postgres())


@lru_cache(maxsize=1)
def get_concept_repository() -> ConceptRepository:
    return ConceptRepository(postgres=get_postgres())


def get_question_entity_linker() -> Any:
    """Assemble the Phase 4 3-tier question entity linker.

    Not cached: the underlying scispacy model is heavyweight and lazy-loaded,
    but we may want multiple QuestionEntityLinker instances with different
    ``min_confidence`` thresholds in API routes / benchmarks.
    """
    from minimed_rag.nlp.linking.entity_linker import QuestionEntityLinker
    from minimed_rag.nlp.linking.scispacy_entity_linker import ScispacyEntityLinker
    from minimed_rag.nlp.linking.synonym_fallback_linker import (
        SynonymFallbackLinker,
    )

    settings = get_settings()
    return QuestionEntityLinker(
        scispacy_linker=ScispacyEntityLinker(
            model_name=settings.scispacy_model,
            use_umls_linker=settings.scispacy_umls_linker,
        ),
        fallback_linker=SynonymFallbackLinker(term_repo=get_term_repository()),
        graph_client=get_graph_client(),
    )


def build_rag_pipeline(use_kg: bool = False) -> Any:
    from minimed_rag.retrieval.rag_pipeline import RAGPipeline

    return RAGPipeline(
        graph_client=get_graph_client() if use_kg else None,
        question_entity_linker=get_question_entity_linker() if use_kg else None,
        hybrid_retriever=None,
        llm=None,
    )
