"""Pseudocode normalize CLI."""

from __future__ import annotations

import typer

app = typer.Typer(help="Normalize terminology and sources")


@app.command()
def terminology(release: str = "local"):
    typer.echo(f"normalize terminology release={release}")


@app.command()
def source(source_name: str):
    typer.echo(f"normalize source={source_name}")
