"""Pseudocode train TRM CLI."""

from __future__ import annotations

import typer

app = typer.Typer(help="Train TRM")


@app.callback(invoke_without_command=True)
def main(config: str = "configs/training/trm_training.yaml", graph_version: str = "kg_local"):
    typer.echo(f"train trm config={config} graph_version={graph_version}")
