"""Build KG CLI."""

from __future__ import annotations

import typer

from minimed_rag.common.config import get_settings
from minimed_rag.kg_build.neo4j_loader import Neo4jLoader
from minimed_rag.kg_build.umls_crosswalk import CrosswalkRunner
from minimed_rag.storage.neo4j import connect as neo4j_connect
from minimed_rag.storage.postgres import connect as postgres_connect

app = typer.Typer(help="Build canonical KG")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context, graph_version: str = "kg_local") -> None:
    ctx.obj = {"graph_version": graph_version}
    if ctx.invoked_subcommand is None:
        typer.echo(f"build kg graph_version={graph_version}")


@app.command("apply-schema")
def apply_schema(
    graph_version: str | None = typer.Option(
        None, "--graph-version", help="Graph version tag (defaults from env/settings)"
    ),
) -> None:
    """Apply Neo4j constraints and indexes for the Phase 4 KG schema."""
    settings = get_settings()
    gv = graph_version or settings.graph_version
    neo4j = neo4j_connect(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        settings.neo4j_database,
    )
    try:
        loader = Neo4jLoader(neo4j, graph_version=gv)
        loader.apply_schema()
        typer.echo(f"Applied Neo4j schema (graph_version={gv})")
    finally:
        neo4j.close()


@app.command("crosswalk")
def crosswalk(
    graph_version: str | None = typer.Option(
        None, "--graph-version", help="Only process entities with this graph_version"
    ),
) -> None:
    """Backfill `Entity.canonical_cui` by matching PrimeKG source_ids to UMLS
    (SAB, CODE) pairs in the Postgres ``umls_crosswalk`` table."""
    settings = get_settings()
    gv = graph_version or settings.graph_version
    neo4j = neo4j_connect(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        settings.neo4j_database,
    )
    postgres = postgres_connect(settings.postgres_dsn)
    try:
        runner = CrosswalkRunner(neo4j=neo4j, postgres=postgres)
        counts = runner.run(graph_version=gv)
        typer.echo(f"crosswalk done: {counts}")
    finally:
        neo4j.close()
        postgres.close()
