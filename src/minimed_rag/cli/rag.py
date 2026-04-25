"""RAG CLI — ask questions; optionally augment with the KG."""

from __future__ import annotations

import json
import logging

import typer

from minimed_rag.common.config import get_settings
from minimed_rag.kg.graph_client import GraphClient
from minimed_rag.nlp.linking.entity_linker import QuestionEntityLinker
from minimed_rag.nlp.linking.scispacy_entity_linker import ScispacyEntityLinker
from minimed_rag.nlp.linking.synonym_fallback_linker import SynonymFallbackLinker
from minimed_rag.retrieval.rag_pipeline import RAGPipeline
from minimed_rag.storage.neo4j import connect as neo4j_connect
from minimed_rag.storage.postgres import connect as postgres_connect
from minimed_rag.storage.repositories.term_repo import TermRepository

app = typer.Typer(help="Ask RAG questions")


def _print_answer(answer, as_json: bool) -> None:
    if as_json:
        payload = {
            "answer": answer.answer,
            "confidence": answer.confidence,
            "linked_mentions": [
                {
                    "mention_text": m.mention_text,
                    "cui": m.cui,
                    "entity_id": m.entity_id,
                    "semantic_type": m.semantic_type,
                    "confidence": m.confidence,
                }
                for m in answer.linked_mentions
            ],
            "graph_paths": [
                {
                    "subject": g.subject_name,
                    "predicate": g.predicate,
                    "object": g.object_name,
                    "confidence": g.confidence,
                    "polarity": g.polarity,
                    "source_system": g.source_system,
                    "assertion_id": g.assertion_id,
                }
                for g in answer.graph_paths
            ],
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo("Linked entities:")
    if not answer.linked_mentions:
        typer.echo("  (none)")
    for m in answer.linked_mentions:
        typer.echo(
            f"  - {m.mention_text} → CUI={m.cui or 'N/A'} "
            f"entity_id={m.entity_id or '-'} "
            f"type={m.semantic_type or '-'} conf={m.confidence:.2f}"
        )

    typer.echo(f"\nGraph paths (top {len(answer.graph_paths)}):")
    if not answer.graph_paths:
        typer.echo("  (none)")
    for g in answer.graph_paths:
        marker = "[-] " if g.polarity == "negative" else "    "
        typer.echo(
            f"  {marker}{g.subject_name} --[{g.predicate}, "
            f"conf={g.confidence:.2f}, src={g.source_system}]--> {g.object_name}"
        )

    typer.echo(f"\nAverage graph confidence: {answer.confidence:.2f}")
    typer.echo(f"\nAnswer:\n{answer.answer}")


@app.command()
def query(
    question: str = typer.Argument(..., help="The biomedical question"),
    use_kg: bool = typer.Option(
        False,
        "--use-kg/--no-use-kg",
        help="Link entities and inject KG neighbors into the prompt",
    ),
    top_k: int = typer.Option(20, "--top-k", help="Max KG facts to inject"),
    per_entity_limit: int = typer.Option(
        10,
        "--per-entity-limit",
        help="Max neighbors per linked entity before global top-k",
    ),
    min_confidence: float = typer.Option(
        0.3,
        "--min-confidence",
        help="Filter out assertions with confidence below this",
    ),
    graph_version: str | None = typer.Option(
        None, "--graph-version", help="Override graph version"
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit structured JSON instead of human-readable output"
    ),
    skip_scispacy_umls_linker: bool = typer.Option(
        False,
        "--skip-scispacy-umls-linker",
        help="Disable scispacy's bundled UMLS linker (fall back to Postgres term match only)",
    ),
) -> None:
    """Ask the RAG pipeline. With ``--use-kg`` it links entities via scispacy
    and injects top-K 1-hop neighbors from Neo4j as structured context."""
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    gv = graph_version or settings.graph_version
    _ = gv  # reserved for per-version retrieval filtering (Phase 5)

    neo4j = neo4j_connect(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        settings.neo4j_database,
    )
    postgres = postgres_connect(settings.postgres_dsn) if use_kg else None
    try:
        graph_client = GraphClient(neo4j=neo4j) if use_kg else None
        question_linker = None
        if use_kg:
            term_repo = TermRepository(postgres=postgres)
            scispacy_linker = ScispacyEntityLinker(
                model_name=settings.scispacy_model,
                use_umls_linker=(settings.scispacy_umls_linker and not skip_scispacy_umls_linker),
            )
            fallback = SynonymFallbackLinker(term_repo=term_repo)
            question_linker = QuestionEntityLinker(
                scispacy_linker=scispacy_linker,
                fallback_linker=fallback,
                graph_client=graph_client,
            )

        pipeline = RAGPipeline(
            graph_client=graph_client,
            question_entity_linker=question_linker,
            hybrid_retriever=None,
            llm=None,
        )
        answer = pipeline.answer(
            question,
            use_kg=use_kg,
            top_k=top_k,
            min_confidence=min_confidence,
            per_entity_limit=per_entity_limit,
        )
        _print_answer(answer, as_json=as_json)
    finally:
        neo4j.close()
        if postgres is not None:
            postgres.close()


@app.command()
def ask(question: str) -> None:
    """Legacy alias — forwards to ``rag query`` without ``--use-kg``."""
    query(
        question=question,
        use_kg=False,
        top_k=20,
        per_entity_limit=10,
        min_confidence=0.3,
        graph_version=None,
        as_json=False,
        skip_scispacy_umls_linker=False,
    )
