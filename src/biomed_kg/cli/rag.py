"""Pseudocode RAG CLI."""
from __future__ import annotations
import typer
app = typer.Typer(help="Ask RAG questions")
@app.command()
def ask(question: str): typer.echo(f"rag ask: {question}")
