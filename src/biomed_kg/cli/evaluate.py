"""Pseudocode evaluate CLI."""
from __future__ import annotations
import typer
app = typer.Typer(help="Evaluate KG-RAG-TRM")
@app.command()
def all(graph_version: str): typer.echo(f"evaluate all graph_version={graph_version}")
