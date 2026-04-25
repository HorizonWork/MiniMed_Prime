"""Build KG CLI."""

from __future__ import annotations

import typer

from minimed_rag.common.config import get_settings
from minimed_rag.kg_build.cypher_conflict_marker import CypherConflictMarker
from minimed_rag.kg_build.graph_stats import GraphStatsBuilder
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
        typer.echo("\nPer-prefix breakdown:")
        typer.echo(runner.prefix_match_rate_summary())
    finally:
        neo4j.close()
        postgres.close()


@app.command("crosswalk-stats")
def crosswalk_stats(
    graph_version: str | None = typer.Option(
        None, "--graph-version",
        help="Only inspect entities with this graph_version (default: all)",
    ),
) -> None:
    """Report ``canonical_cui`` coverage on existing :Entity nodes.

    Use after ``crosswalk`` to diagnose low KG hit rates in benchmarks.
    Coverage <50% on key prefixes (MONDO, DRUGBANK, NCBI gene IDs) usually
    means MRCONSO is missing the relevant SABs — re-ingest with the right
    UMLS slice or extend ``_source_id_to_sab_code``.
    """
    settings = get_settings()
    neo4j = neo4j_connect(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        settings.neo4j_database,
    )
    try:
        cypher = (
            "MATCH (e:Entity) "
            + ("WHERE e.graph_version = $gv " if graph_version else "")
            + "RETURN e.entity_type AS entity_type, "
            "       e.source_id AS source_id, "
            "       e.canonical_cui IS NOT NULL AND e.canonical_cui <> '' AS has_cui"
        )
        params = {"gv": graph_version} if graph_version else {}
        rows = neo4j.query(cypher, params)
        if not rows:
            typer.echo("No :Entity nodes found.")
            return

        from minimed_rag.kg_build.umls_crosswalk import _prefix_of

        by_prefix: dict[str, dict[str, int]] = {}
        by_type: dict[str, dict[str, int]] = {}
        total = 0
        with_cui = 0
        for row in rows:
            total += 1
            prefix = _prefix_of(row.get("source_id", "") or "")
            etype = row.get("entity_type") or "(unknown)"
            has = bool(row.get("has_cui"))
            with_cui += 1 if has else 0
            p_bucket = by_prefix.setdefault(prefix, {"total": 0, "linked": 0})
            t_bucket = by_type.setdefault(etype, {"total": 0, "linked": 0})
            p_bucket["total"] += 1
            t_bucket["total"] += 1
            if has:
                p_bucket["linked"] += 1
                t_bucket["linked"] += 1

        rate = with_cui / total if total else 0.0
        typer.echo(f"Total :Entity nodes: {total}")
        typer.echo(f"With canonical_cui:  {with_cui}  ({rate:.1%})")

        typer.echo(f"\n{'prefix':<14} {'total':>10} {'linked':>10} {'rate':>8}")
        for prefix, bucket in sorted(by_prefix.items(), key=lambda kv: kv[1]["total"], reverse=True):
            r = bucket["linked"] / bucket["total"] if bucket["total"] else 0.0
            typer.echo(f"{prefix:<14} {bucket['total']:>10d} {bucket['linked']:>10d} {r:>7.1%}")

        typer.echo(f"\n{'entity_type':<24} {'total':>10} {'linked':>10} {'rate':>8}")
        for etype, bucket in sorted(by_type.items(), key=lambda kv: kv[1]["total"], reverse=True):
            r = bucket["linked"] / bucket["total"] if bucket["total"] else 0.0
            typer.echo(f"{etype:<24} {bucket['total']:>10d} {bucket['linked']:>10d} {r:>7.1%}")
    finally:
        neo4j.close()


@app.command("compute-graph-stats")
def compute_graph_stats(
    snapshot_id: str | None = typer.Option(
        None, "--snapshot-id",
        help="Tag for this stats snapshot (defaults to graph_version).",
    ),
) -> None:
    """Compute per-entity assertion-degree counts and persist to Postgres
    ``entity_stats``. Used by PathScorer (hub_penalty) and NegativeSampler
    (generate_generic_hub_paths). Idempotent per snapshot_id — re-running
    replaces the snapshot."""
    settings = get_settings()
    sid = snapshot_id or settings.graph_version
    neo4j = neo4j_connect(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        settings.neo4j_database,
    )
    postgres = postgres_connect(settings.postgres_dsn)
    try:
        builder = GraphStatsBuilder(neo4j=neo4j, postgres=postgres)
        counts = builder.compute_and_persist(snapshot_id=sid)
        typer.echo(f"compute-graph-stats snapshot={sid}: {counts}")
    finally:
        neo4j.close()
        postgres.close()


@app.command("detect-conflicts")
def detect_conflicts(
    graph_version: str | None = typer.Option(
        None, "--graph-version",
        help="Only mark assertions in this graph_version (default: all).",
    ),
    no_reset: bool = typer.Option(
        False, "--no-reset",
        help="Skip zeroing existing conflict_score before re-marking.",
    ),
) -> None:
    """Stamp ``Assertion.conflict_score`` for self-conflicting facts.

    Two patterns: (1) same (subject, predicate, object) with opposite
    polarities → 0.5; (2) named conflicting predicate pairs (e.g.,
    ``treats`` vs ``contraindicated_for``) on the same (subject, object)
    → +0.25 each, capped at 1.0.

    Required by ``generate-paths --include-negatives`` so the
    ``unsupported_kg_conflict`` negative-path generator has signal to
    work with. Idempotent (resets first by default).
    """
    settings = get_settings()
    neo4j = neo4j_connect(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        settings.neo4j_database,
    )
    try:
        marker = CypherConflictMarker(neo4j=neo4j)
        counts = marker.run(graph_version=graph_version, reset_first=not no_reset)
        typer.echo(f"detect-conflicts: {counts}")
    finally:
        neo4j.close()
