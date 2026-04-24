"""Pseudocode ingest CLI."""

from __future__ import annotations

import typer

app = typer.Typer(help="Ingest biomedical sources")


@app.command()
def source(source_name: str, release: str = "local", mode: str = "baseline"):
    typer.echo(f"ingest source={source_name} release={release} mode={mode}")
