from __future__ import annotations

import typer

app = typer.Typer(help="Build serving indexes")

_CORPUS_CHOICES = ("pubmed", "textbooks", "statpearls")
_DEFAULT_DENSE_INDEX = "artifacts/indexes/dense_textbooks_bge-small-en-v1.5"


@app.command()
def graph(graph_version: str):
    typer.echo(f"build graph index graph_version={graph_version}")


@app.command()
def vector(graph_version: str):
    typer.echo(f"build vector index graph_version={graph_version}")


@app.command()
def search(graph_version: str):
    typer.echo(f"build search index graph_version={graph_version}")


@app.command()
def all(graph_version: str):
    typer.echo(f"build all indexes graph_version={graph_version}")


@app.command()
def bm25(
    corpus: list[str] = typer.Option(
        ["textbooks"],
        help=f"Corpus sources to include. Choices: {', '.join(_CORPUS_CHOICES)}",
    ),
    limit: int = typer.Option(
        10_000,
        help="Max chunks per corpus source (0 = no limit)",
    ),
    output: str = typer.Option(
        "artifacts/indexes/bm25_textbooks.pkl",
        help="Output path for the index pickle",
    ),
):
    """Build a standalone BM25 index from HuggingFace corpus datasets."""
    from pathlib import Path

    from minimed_rag.index.bm25_index import BM25Index

    chunks = _load_corpus_chunks(corpus, limit)

    typer.echo(f"\nBuilding BM25 index over {len(chunks):,} chunks …")
    idx = BM25Index()
    idx.build(chunks)

    out_path = Path(output)
    idx.save(out_path)
    typer.echo(f"Index saved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


@app.command()
def dense(
    corpus: list[str] = typer.Option(
        ["textbooks"],
        help=f"Corpus sources to include. Choices: {', '.join(_CORPUS_CHOICES)}",
    ),
    limit: int = typer.Option(
        10_000,
        help="Max chunks per corpus source (0 = no limit)",
    ),
    output: str = typer.Option(
        _DEFAULT_DENSE_INDEX,
        help="Output directory for the dense index artifact",
    ),
    model_name: str = typer.Option(
        "BAAI/bge-small-en-v1.5",
        help="Embedding model id",
    ),
    batch_size: int = typer.Option(32, help="Embedding batch size"),
):
    """Build a standalone dense index from HuggingFace corpus datasets."""
    from pathlib import Path

    from minimed_rag.index.embedding import TransformerEmbedder
    from minimed_rag.index.faiss_index import FaissIndex

    chunks = _load_corpus_chunks(corpus, limit)
    typer.echo(f"\nEncoding {len(chunks):,} chunks with {model_name} …")
    embedder = TransformerEmbedder(model_name=model_name, batch_size=batch_size)
    embeddings = embedder.encode([chunk.text for chunk in chunks])

    typer.echo(f"Building dense index dim={embeddings.shape[1]} …")
    out_path = Path(output)
    idx = FaissIndex(
        path=out_path,
        metadata={
            "model_name": model_name,
            "corpus": list(corpus),
        },
    )
    idx.build(chunks, embeddings)
    idx.save()
    typer.echo(f"Dense index saved → {out_path}")


def _load_corpus_chunks(corpus: list[str], limit: int):
    from minimed_rag.corpus.pubmed_loader import PubMedLoader
    from minimed_rag.corpus.statpearls_loader import StatPearlsLoader
    from minimed_rag.corpus.textbooks_loader import TextbooksLoader

    loaders = {
        "pubmed": PubMedLoader,
        "textbooks": TextbooksLoader,
        "statpearls": StatPearlsLoader,  # raises RuntimeError with clear message
    }

    invalid = [c for c in corpus if c not in loaders]
    if invalid:
        typer.echo(f"Unknown corpus: {invalid}. Choices: {list(loaders)}", err=True)
        raise typer.Exit(1)

    chunks = []
    for name in corpus:
        loader = loaders[name]()
        typer.echo(f"Loading {name} (limit={limit or 'none'}) …")
        loaded = loader.load(limit=limit)
        typer.echo(f"  {len(loaded):,} chunks")
        chunks.extend(loaded)
    return chunks
