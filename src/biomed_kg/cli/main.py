"""Pseudocode biomedkg CLI entrypoint."""
from __future__ import annotations

import typer

from biomed_kg.cli import build_indexes, build_kg, evaluate, generate_paths, ingest, normalize, rag, train_trm

app = typer.Typer(help="Biomedical KG-RAG CLI")
app.add_typer(ingest.app, name="ingest")
app.add_typer(normalize.app, name="normalize")
app.add_typer(build_kg.app, name="build-kg")
app.add_typer(build_indexes.app, name="build-indexes")
app.add_typer(rag.app, name="rag")
app.add_typer(generate_paths.app, name="generate-paths")
app.add_typer(train_trm.app, name="train-trm")
app.add_typer(evaluate.app, name="evaluate")


if __name__ == "__main__":
    app()
