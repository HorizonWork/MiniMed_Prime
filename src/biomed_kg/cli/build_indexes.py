"""Pseudocode build indexes CLI."""
from __future__ import annotations
import typer
app = typer.Typer(help="Build serving indexes")
@app.command()
def graph(graph_version: str): typer.echo(f"build graph index graph_version={graph_version}")
@app.command()
def vector(graph_version: str): typer.echo(f"build vector index graph_version={graph_version}")
@app.command()
def search(graph_version: str): typer.echo(f"build search index graph_version={graph_version}")
@app.command()
def all(graph_version: str): typer.echo(f"build all indexes graph_version={graph_version}")
