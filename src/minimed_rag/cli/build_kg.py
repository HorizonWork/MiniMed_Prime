"""Pseudocode build KG CLI."""

from __future__ import annotations

import typer

app = typer.Typer(help="Build canonical KG")


@app.callback(invoke_without_command=True)
def main(graph_version: str = "kg_local"):
    typer.echo(f"build kg graph_version={graph_version}")
