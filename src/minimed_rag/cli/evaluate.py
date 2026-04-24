from __future__ import annotations

import typer

app = typer.Typer(help="Evaluate KG-RAG-TRM quality")

_DEFAULT_BM25_INDEX = "artifacts/indexes/bm25_textbooks.pkl"
_DEFAULT_DENSE_INDEX = "artifacts/indexes/dense_textbooks_bge-small-en-v1.5"


@app.command()
def all(graph_version: str):
    """Evaluate end-to-end KG-RAG pipeline quality."""
    typer.echo(f"evaluate all graph_version={graph_version}")


@app.command()
def benchmark(
    suite: str = typer.Option(
        "mirage", help="Suite: mirage | mmlu | medqa | medmcqa | pubmedqa | bioasq"
    ),
    limit: int = typer.Option(50, help="Max examples per dataset (0 = no limit)"),
    model: str = typer.Option(
        "random",
        help="Model spec: random | local | ollama:<model> | hf:<model-id>",
    ),
    retriever: str = typer.Option(
        "none",
        help="Retriever: none | bm25 | dense | hybrid | hybrid_rerank",
    ),
    index_path: str = typer.Option(
        _DEFAULT_BM25_INDEX,
        help="Path to BM25 index pickle",
    ),
    dense_index_path: str = typer.Option(
        _DEFAULT_DENSE_INDEX,
        help="Path to dense index artifact directory",
    ),
    embedding_model: str | None = typer.Option(
        None,
        help="Embedding model override for dense retrieval",
    ),
    reranker_model: str = typer.Option(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder reranker model for --retriever hybrid_rerank",
    ),
    retrieval_config: str | None = typer.Option(
        None,
        help="Optional retrieval ablation config YAML",
    ),
    top_k: int = typer.Option(10, help="Number of chunks to retrieve per question"),
    output_dir: str = typer.Option(
        "artifacts/reports", help="Directory for JSONL and Markdown reports"
    ),
    split: str = typer.Option("test", help="Dataset split to use"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print per-example progress"),
):
    """Run a MIRAGE-style benchmark and write accuracy / latency reports."""
    from minimed_rag.benchmark.datasets.bioasq_yn import BioASQDataset
    from minimed_rag.benchmark.datasets.medmcqa import MedMCQADataset
    from minimed_rag.benchmark.datasets.medqa_us import MedQADataset
    from minimed_rag.benchmark.datasets.mmlu_med import MMLUMedDataset
    from minimed_rag.benchmark.datasets.pubmedqa import PubMedQADataset
    from minimed_rag.benchmark.evaluator import run_evaluation

    _SUITES: dict[str, list] = {
        "mirage": [
            MMLUMedDataset(),
            MedQADataset(),
            MedMCQADataset(),
            PubMedQADataset(),
            BioASQDataset(),
        ],
        "mmlu": [MMLUMedDataset()],
        "medqa": [MedQADataset()],
        "medmcqa": [MedMCQADataset()],
        "pubmedqa": [PubMedQADataset()],
        "bioasq": [BioASQDataset()],
    }

    if suite not in _SUITES:
        typer.echo(f"Unknown suite '{suite}'. Available: {', '.join(_SUITES)}", err=True)
        raise typer.Exit(1)

    if retrieval_config is not None:
        overrides = _load_retrieval_overrides(retrieval_config)
        retriever = overrides.get("retriever", retriever)
        top_k = int(overrides.get("top_k", top_k))
        index_path = overrides.get("index_path", index_path)
        dense_index_path = overrides.get("dense_index_path", dense_index_path)
        embedding_model = overrides.get("embedding_model", embedding_model)
        reranker_model = overrides.get("reranker_model", reranker_model)

    provider = _build_provider(model)
    typer.echo(f"Provider: {provider.name}")

    retrieval = _build_retriever(
        retriever,
        index_path=index_path,
        dense_index_path=dense_index_path,
        top_k=top_k,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
    )
    if retrieval is not None:
        typer.echo(f"Retriever: {retriever}  top_k={top_k}")

    examples = []
    for ds in _SUITES[suite]:
        typer.echo(f"Loading {ds.name}...")
        try:
            loaded = ds.load(split=split)
            if limit > 0:
                loaded = loaded[:limit]
            examples.extend(loaded)
            typer.echo(f"  {len(loaded)} examples")
        except Exception as exc:
            typer.echo(f"  WARNING: could not load {ds.name}: {exc}", err=True)

    if not examples:
        typer.echo("No examples loaded — check dataset availability.", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nEvaluating {len(examples)} examples with {provider.name} …")
    report = run_evaluation(
        suite, examples, provider, output_dir, retriever=retrieval, verbose=verbose
    )

    typer.echo(f"\n{'─' * 50}")
    typer.echo(f"Suite        {report.suite}")
    typer.echo(f"Model        {report.model}")
    typer.echo(f"Config hash  {report.config_hash}")
    typer.echo(f"Total        {report.total}")
    typer.echo(f"Accuracy     {report.accuracy:.1%}")
    typer.echo(f"Invalid      {report.invalid_rate:.1%}  ({report.invalid} examples)")
    typer.echo(f"Latency      {report.mean_latency_s:.3f}s mean")

    if report.retrieval_diagnostics:
        d = report.retrieval_diagnostics
        typer.echo(f"\nRetrieval diagnostics ({d.retriever}):")
        typer.echo(f"  Corpus hit rate  {d.corpus_hit_rate:.1%}")
        typer.echo(f"  Avg ret score    {d.avg_score:.3f}")
        typer.echo(f"  Avg ret latency  {d.avg_latency_s:.3f}s")

    typer.echo("")
    typer.echo("Per-dataset:")
    for ds_name, stats in report.per_dataset.items():
        typer.echo(
            f"  {ds_name:<20} {stats['correct']:>4}/{stats['total']:<4}"
            f"  acc={stats['accuracy']:.1%}  invalid={stats['invalid']}"
        )


def _build_provider(model: str):
    from minimed_rag.benchmark.providers.hf_local import HuggingFaceProvider
    from minimed_rag.benchmark.providers.ollama import OllamaProvider
    from minimed_rag.benchmark.providers.random_baseline import RandomBaseline

    if model in ("random", "stub"):
        return RandomBaseline()

    if model == "local":
        return OllamaProvider(model="qwen2.5:7b-instruct")

    if model.startswith("ollama:"):
        return OllamaProvider(model=model[len("ollama:") :])

    if model.startswith("hf:"):
        return HuggingFaceProvider(model_id=model[len("hf:") :])

    typer.echo(
        f"Unknown model spec '{model}'.\n  Use: random | local | ollama:<model> | hf:<model-id>",
        err=True,
    )
    raise typer.Exit(1)


def _build_retriever(
    retriever: str,
    *,
    index_path: str,
    dense_index_path: str,
    top_k: int,
    embedding_model: str | None,
    reranker_model: str,
):
    from pathlib import Path

    if retriever == "none":
        return None

    if retriever == "bm25":
        p = _require_bm25_index(Path(index_path))
        from minimed_rag.retrieval.bm25_retriever import LocalBM25Retriever

        return LocalBM25Retriever(index_path=p, top_k=top_k)

    if retriever == "dense":
        p = _require_dense_index(Path(dense_index_path))
        from minimed_rag.retrieval.vector_retriever import LocalDenseRetriever

        return LocalDenseRetriever(index_path=p, model_name=embedding_model, top_k=top_k)

    if retriever in {"hybrid", "hybrid_rerank"}:
        bm25_path = _require_bm25_index(Path(index_path))
        dense_path = _require_dense_index(Path(dense_index_path))

        from minimed_rag.retrieval.bm25_retriever import LocalBM25Retriever
        from minimed_rag.retrieval.hybrid_retriever import HybridRetriever
        from minimed_rag.retrieval.vector_retriever import LocalDenseRetriever

        reranker = None
        if retriever == "hybrid_rerank":
            from minimed_rag.retrieval.reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker(model_name=reranker_model)

        return HybridRetriever(
            bm25_retriever=LocalBM25Retriever(index_path=bm25_path, top_k=top_k),
            dense_retriever=LocalDenseRetriever(
                index_path=dense_path,
                model_name=embedding_model,
                top_k=top_k,
            ),
            top_k=top_k,
            bm25_k=max(50, top_k),
            dense_k=max(50, top_k),
            reranker=reranker,
        )

    typer.echo(
        f"Unknown retriever '{retriever}'. Use: none | bm25 | dense | hybrid | hybrid_rerank",
        err=True,
    )
    raise typer.Exit(1)


def _load_retrieval_overrides(config_path: str) -> dict:
    from minimed_rag.common.config import load_yaml

    data = load_yaml(config_path)
    retrieval = data.get("retrieval", data)
    dense = retrieval.get("dense", {})
    bm25 = retrieval.get("bm25", {})
    reranker = retrieval.get("reranker", {})

    overrides = {
        "retriever": retrieval.get("retriever"),
        "top_k": retrieval.get("top_k"),
        "index_path": bm25.get("index_path"),
        "dense_index_path": dense.get("index_path"),
        "embedding_model": dense.get("model_name"),
        "reranker_model": reranker.get("model_name"),
    }
    return {key: value for key, value in overrides.items() if value is not None}


def _require_bm25_index(path):
    if path.exists():
        return path
    typer.echo(
        f"BM25 index not found at '{path}'.\n"
        "  Build it first:\n"
        "    uv run minimed build-indexes bm25 --corpus textbooks --limit 10000",
        err=True,
    )
    raise typer.Exit(1)


def _require_dense_index(path):
    if (path / "vectors.npy").exists() and (path / "chunks.pkl").exists():
        return path
    typer.echo(
        f"Dense index not found at '{path}'.\n"
        "  Build it first:\n"
        "    uv run minimed build-indexes dense --corpus textbooks --limit 10000",
        err=True,
    )
    raise typer.Exit(1)
