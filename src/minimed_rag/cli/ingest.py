"""Ingest CLI — dispatch per-source pipelines."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from minimed_rag.common.config import get_settings
from minimed_rag.ingestion.primekg.downloader import PrimeKGDownloader
from minimed_rag.ingestion.primekg.pipeline import PrimeKGIngestionPipeline
from minimed_rag.ingestion.primekg.predicate_map import SimplePredicateMapper
from minimed_rag.ingestion.umls.pipeline import UMLSIngestionPipeline
from minimed_rag.ingestion.umls.predicate_map import UMLSPredicateMapper
from minimed_rag.kg_build.neo4j_loader import Neo4jLoader
from minimed_rag.schema_registry.predicate_registry import PredicateRegistry
from minimed_rag.storage.neo4j import connect as neo4j_connect
from minimed_rag.storage.postgres import connect as postgres_connect

logger = logging.getLogger(__name__)

app = typer.Typer(help="Ingest biomedical sources")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command()
def primekg(
    release: str = typer.Option("local", "--release", help="Release tag"),
    force: bool = typer.Option(False, "--force", help="Redownload even if file exists"),
    limit: int | None = typer.Option(None, "--limit", help="Stop after N CSV rows (smoke testing)"),
    graph_version: str | None = typer.Option(
        None, "--graph-version", help="Override graph version"
    ),
    apply_schema: bool = typer.Option(
        True,
        "--apply-schema/--skip-schema",
        help="Apply Neo4j constraints + indexes before ingest",
    ),
) -> None:
    """Ingest PrimeKG edges into Neo4j (reified :Assertion schema)."""
    _configure_logging()
    settings = get_settings()
    gv = graph_version or settings.graph_version

    downloader = PrimeKGDownloader(
        data_dir=settings.primekg_data_dir,
        url=settings.primekg_download_url,
        force=force,
    )
    neo4j = neo4j_connect(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        settings.neo4j_database,
    )
    try:
        loader = Neo4jLoader(neo4j, graph_version=gv)
        if apply_schema:
            loader.apply_schema()
        pipeline = PrimeKGIngestionPipeline(
            downloader=downloader,
            predicate_mapper=SimplePredicateMapper(),
            neo4j_loader=loader,
            graph_version=gv,
            predicate_registry=PredicateRegistry(),
        )
        counts = pipeline.run(release=release, limit=limit)
        typer.echo(f"primekg ingest done: {counts}")
    finally:
        neo4j.close()


@app.command()
def umls(
    release: str = typer.Option("local", "--release", help="Release tag"),
    data_dir: Path | None = typer.Option(
        None, "--data-dir", help="Override UMLS data dir (default from settings)"
    ),
    graph_version: str | None = typer.Option(
        None, "--graph-version", help="Override graph version"
    ),
    apply_schema: bool = typer.Option(
        True,
        "--apply-schema/--skip-schema",
        help="Apply Neo4j constraints + indexes before ingest",
    ),
    skip_postgres: bool = typer.Option(
        False,
        "--skip-postgres",
        help="Skip Postgres writes (use when schema not bootstrapped)",
    ),
    skip: list[str] = typer.Option(
        [],
        "--skip",
        help="Skip stages: mrconso, mrsty, or mrrel (repeatable)",
    ),
    limit_mrconso: int | None = typer.Option(
        None, "--limit-mrconso", help="Smoke-test limit for MRCONSO rows"
    ),
    limit_mrsty: int | None = typer.Option(
        None, "--limit-mrsty", help="Smoke-test limit for MRSTY rows"
    ),
    limit_mrrel: int | None = typer.Option(
        None, "--limit-mrrel", help="Smoke-test limit for MRREL rows"
    ),
) -> None:
    """Ingest UMLS MRCONSO / MRSTY / MRREL into Neo4j (+ Postgres)."""
    _configure_logging()
    settings = get_settings()
    gv = graph_version or settings.graph_version
    data_root = Path(data_dir) if data_dir else Path(settings.umls_data_dir)

    neo4j = neo4j_connect(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        settings.neo4j_database,
    )
    postgres = None if skip_postgres else postgres_connect(settings.postgres_dsn)
    try:
        loader = Neo4jLoader(neo4j, graph_version=gv)
        if apply_schema:
            loader.apply_schema()
        pipeline = UMLSIngestionPipeline(
            neo4j_loader=loader,
            postgres=postgres,
            predicate_mapper=UMLSPredicateMapper(),
            data_dir=data_root,
            graph_version=gv,
        )
        counts = pipeline.run(
            release=release,
            limit_mrconso=limit_mrconso,
            limit_mrsty=limit_mrsty,
            limit_mrrel=limit_mrrel,
            skip=skip,
        )
        typer.echo(f"umls ingest done: {counts}")
    finally:
        neo4j.close()
        if postgres is not None:
            postgres.close()


@app.command()
def source(
    source_name: str,
    release: str = typer.Option("local", "--release"),
    mode: str = typer.Option("baseline", "--mode", help="Legacy flag, ignored for primekg/umls"),
) -> None:
    """Legacy dispatcher. Prefer ``minimed ingest primekg`` / ``minimed ingest umls``."""
    if source_name == "primekg":
        primekg(
            release=release,
            force=False,
            limit=None,
            graph_version=None,
            apply_schema=True,
        )
    elif source_name == "umls":
        umls(
            release=release,
            data_dir=None,
            graph_version=None,
            apply_schema=True,
            skip_postgres=False,
            skip=[],
            limit_mrconso=None,
            limit_mrsty=None,
            limit_mrrel=None,
        )
    else:
        typer.echo(f"ingest source={source_name} release={release} mode={mode}")
