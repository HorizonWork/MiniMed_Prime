"""Pseudocode generate paths CLI."""
from __future__ import annotations
import typer
app = typer.Typer(help="Generate reasoning paths")
@app.callback(invoke_without_command=True)
def main(dataset: str = "medreason", graph_version: str = "kg_local", split: str = "train"):
    typer.echo(f"generate paths dataset={dataset} graph_version={graph_version} split={split}")
