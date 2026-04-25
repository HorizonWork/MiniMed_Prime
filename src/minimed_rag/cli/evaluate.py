from __future__ import annotations

import logging

import typer

app = typer.Typer(help="Evaluate KG-RAG-TRM quality")

_DEFAULT_BM25_INDEX = "artifacts/indexes/bm25_textbooks.pkl"
_DEFAULT_DENSE_INDEX = "artifacts/indexes/dense_textbooks_bge-small-en-v1.5"
_DEFAULT_REPORT_DIR = "artifacts/reports"

logger = logging.getLogger(__name__)


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
    max_context_chars: int = typer.Option(
        2500, "--max-context-chars", help="Max characters of retrieval context in prompts"
    ),
    graph_budget_ratio: float = typer.Option(
        0.4, "--graph-budget-ratio", help="Share of context budget reserved for KG paths"
    ),
    output_dir: str = typer.Option(
        _DEFAULT_REPORT_DIR, help="Directory for JSONL and Markdown reports"
    ),
    split: str = typer.Option("test", help="Dataset split to use"),
    use_kg: bool = typer.Option(
        False, "--use-kg/--no-use-kg",
        help="Augment retrieval with KG paths (1- and 2-hop) from Neo4j",
    ),
    kg_min_confidence: float = typer.Option(
        0.5, "--kg-min-confidence",
        help="Drop graph assertions with confidence below this",
    ),
    kg_hops: int = typer.Option(
        2, "--kg-hops", help="Max hops to expand from each anchor (1 or 2)",
    ),
    kg_max_paths: int = typer.Option(
        20, "--kg-max-paths", help="Max graph paths to inject per question",
    ),
    kg_per_entity_limit: int = typer.Option(
        10, "--kg-per-entity-limit",
        help="Max neighbors fetched per linked entity before global cap",
    ),
    ablation: bool = typer.Option(
        False, "--ablation/--no-ablation",
        help="Run text-only / kg-only / kg+text comparison (--use-kg enables this)",
    ),
    report_conflicts: bool = typer.Option(
        True, "--report-conflicts/--no-report-conflicts",
        help="Detect and log graph↔text conflicts when KG is active",
    ),
    use_chat_template: bool = typer.Option(
        True, "--use-chat-template/--no-use-chat-template",
        help="Apply the model's chat template (HF provider only; no-op for others)",
    ),
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
        max_context_chars = int(overrides.get("max_context_chars", max_context_chars))
        graph_budget_ratio = float(overrides.get("graph_budget_ratio", graph_budget_ratio))
        index_path = overrides.get("index_path", index_path)
        dense_index_path = overrides.get("dense_index_path", dense_index_path)
        embedding_model = overrides.get("embedding_model", embedding_model)
        reranker_model = overrides.get("reranker_model", reranker_model)
        kg_min_confidence = float(overrides.get("kg_min_confidence", kg_min_confidence))
        kg_hops = int(overrides.get("kg_hops", kg_hops))
        kg_max_paths = int(overrides.get("kg_max_paths", kg_max_paths))
        kg_per_entity_limit = int(overrides.get("kg_per_entity_limit", kg_per_entity_limit))
        if "use_kg" in overrides:
            use_kg = bool(overrides["use_kg"])

    if use_kg:
        ablation = True

    provider = _build_provider(model, use_chat_template=use_chat_template)
    typer.echo(f"Provider: {provider.name}")

    examples = _load_examples(_SUITES[suite], split=split, limit=limit)
    if not examples:
        typer.echo("No examples loaded — check dataset availability.", err=True)
        raise typer.Exit(1)

    # Build text retriever (may be None for retriever='none')
    text_retriever = _build_text_retriever(
        retriever,
        index_path=index_path,
        dense_index_path=dense_index_path,
        top_k=top_k,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
    )
    if text_retriever is not None:
        typer.echo(f"Retriever: {retriever}  top_k={top_k}")

    # Build KG components (graceful degrade if Neo4j unreachable)
    graph_components = None
    if use_kg or ablation:
        graph_components = _build_graph_components(
            kg_min_confidence=kg_min_confidence,
            kg_hops=kg_hops,
            kg_max_paths=kg_max_paths,
            kg_per_entity_limit=kg_per_entity_limit,
        )
        if graph_components is None:
            typer.echo("WARNING: KG components unavailable — degrading to text-only.", err=True)
            use_kg = False
            ablation = False

    conflict_reporter = None
    conflict_dir = None
    if (use_kg or ablation) and report_conflicts and graph_components is not None:
        from minimed_rag.kg_build.runtime_conflict_reporter import RuntimeConflictReporter

        conflict_reporter = RuntimeConflictReporter()
        conflict_dir = output_dir

    try:
        if ablation and graph_components is not None:
            _run_ablation(
                suite=suite,
                examples=examples,
                provider=provider,
                output_dir=output_dir,
                text_retriever=text_retriever,
                graph_components=graph_components,
                conflict_reporter=conflict_reporter,
                conflict_dir=conflict_dir,
                max_context_chars=max_context_chars,
                graph_budget_ratio=graph_budget_ratio,
                verbose=verbose,
            )
        else:
            kg_retriever = None
            if use_kg and graph_components is not None:
                kg_retriever = _build_kg_augmented_retriever(
                    text_retriever=text_retriever,
                    graph_components=graph_components,
                    text_top_k=top_k,
                    name="kg_text" if text_retriever is not None else "kg_only",
                )

            report = run_evaluation(
                suite, examples, provider, output_dir,
                retriever=text_retriever if kg_retriever is None else None,
                kg_retriever=kg_retriever,
                conflict_reporter=conflict_reporter,
                conflict_output_dir=conflict_dir,
                max_context_chars=max_context_chars,
                graph_budget_ratio=graph_budget_ratio,
                verbose=verbose,
            )
            _print_report_summary(report)
    finally:
        if graph_components is not None:
            try:
                graph_components.close()
            except Exception:
                pass


def _load_examples(datasets, *, split: str, limit: int) -> list:
    examples = []
    for ds in datasets:
        typer.echo(f"Loading {ds.name}...")
        try:
            loaded = ds.load(split=split)
            if limit > 0:
                loaded = loaded[:limit]
            examples.extend(loaded)
            typer.echo(f"  {len(loaded)} examples")
        except Exception as exc:
            typer.echo(f"  WARNING: could not load {ds.name}: {exc}", err=True)
    return examples


def _print_report_summary(report) -> None:
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
        if d.kg_hit_rate is not None:
            typer.echo(f"  KG hit rate      {d.kg_hit_rate:.1%}")
            typer.echo(f"  Avg KG paths     {d.avg_kg_paths or 0.0:.2f}")
            typer.echo(f"  Avg KG latency   {d.avg_kg_latency_s or 0.0:.3f}s")
        if d.n_conflicts is not None:
            typer.echo(f"  Conflicts        {d.n_conflicts}")

    typer.echo("")
    typer.echo("Per-dataset:")
    for ds_name, stats in report.per_dataset.items():
        typer.echo(
            f"  {ds_name:<20} {stats['correct']:>4}/{stats['total']:<4}"
            f"  acc={stats['accuracy']:.1%}  invalid={stats['invalid']}"
        )


def _run_ablation(
    *,
    suite: str,
    examples: list,
    provider,
    output_dir: str,
    text_retriever,
    graph_components,
    conflict_reporter,
    conflict_dir,
    max_context_chars: int,
    graph_budget_ratio: float,
    verbose: bool,
) -> None:
    from minimed_rag.benchmark.ablation_report import write_ablation_report
    from minimed_rag.benchmark.evaluator import run_evaluation

    reports = {}

    typer.echo("\n=== Ablation: text-only ===")
    text_only_report = run_evaluation(
        suite, examples, provider, output_dir,
        retriever=text_retriever,
        kg_retriever=None,
        conflict_reporter=None,
        conflict_output_dir=None,
        config_label="text_only",
        max_context_chars=max_context_chars,
        graph_budget_ratio=graph_budget_ratio,
        verbose=verbose,
    )
    reports["text_only"] = text_only_report
    _print_report_summary(text_only_report)

    typer.echo("\n=== Ablation: kg-only ===")
    kg_only_retriever = _build_kg_augmented_retriever(
        text_retriever=None,
        graph_components=graph_components,
        text_top_k=None,
        name="kg_only",
    )
    kg_only_report = run_evaluation(
        suite, examples, provider, output_dir,
        retriever=None,
        kg_retriever=kg_only_retriever,
        conflict_reporter=None,
        conflict_output_dir=None,
        config_label="kg_only",
        max_context_chars=max_context_chars,
        graph_budget_ratio=graph_budget_ratio,
        verbose=verbose,
    )
    reports["kg_only"] = kg_only_report
    _print_report_summary(kg_only_report)

    if text_retriever is not None:
        typer.echo("\n=== Ablation: kg+text ===")
        hybrid_retriever = _build_kg_augmented_retriever(
            text_retriever=text_retriever,
            graph_components=graph_components,
            text_top_k=None,
            name="kg_text",
        )
        hybrid_report = run_evaluation(
            suite, examples, provider, output_dir,
            retriever=None,
            kg_retriever=hybrid_retriever,
            conflict_reporter=conflict_reporter,
            conflict_output_dir=conflict_dir,
            config_label="kg_text",
            max_context_chars=max_context_chars,
            graph_budget_ratio=graph_budget_ratio,
            verbose=verbose,
        )
        reports["kg_text"] = hybrid_report
        _print_report_summary(hybrid_report)

    md_path = write_ablation_report(reports, output_dir, suite)
    typer.echo(f"\nAblation report: {md_path}")


def _build_provider(model: str, *, use_chat_template: bool = True):
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
        return HuggingFaceProvider(
            model_id=model[len("hf:") :],
            use_chat_template=use_chat_template,
        )

    typer.echo(
        f"Unknown model spec '{model}'.\n  Use: random | local | ollama:<model> | hf:<model-id>",
        err=True,
    )
    raise typer.Exit(1)


def _build_text_retriever(
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


class _GraphComponents:
    """Bundle of (graph_client, question_linker, graph_retriever, neo4j, postgres) — handles teardown."""

    def __init__(
        self,
        *,
        graph_client,
        question_linker,
        graph_retriever_factory,
        neo4j,
        postgres,
    ) -> None:
        self.graph_client = graph_client
        self.question_linker = question_linker
        self.graph_retriever_factory = graph_retriever_factory
        self._neo4j = neo4j
        self._postgres = postgres

    def make_graph_retriever(self):
        return self.graph_retriever_factory()

    def close(self) -> None:
        if self._neo4j is not None:
            try:
                self._neo4j.close()
            except Exception:
                pass
        if self._postgres is not None:
            try:
                self._postgres.close()
            except Exception:
                pass


def _build_graph_components(
    *,
    kg_min_confidence: float,
    kg_hops: int,
    kg_max_paths: int,
    kg_per_entity_limit: int,
) -> _GraphComponents | None:
    """Construct GraphClient + QuestionEntityLinker + GraphRetriever factory.

    Returns ``None`` when Neo4j or required dependencies are unavailable;
    caller logs and degrades to text-only.
    """
    try:
        from minimed_rag.common.config import get_settings
        from minimed_rag.kg.graph_client import GraphClient
        from minimed_rag.nlp.linking.entity_linker import QuestionEntityLinker
        from minimed_rag.nlp.linking.scispacy_entity_linker import ScispacyEntityLinker
        from minimed_rag.nlp.linking.synonym_fallback_linker import SynonymFallbackLinker
        from minimed_rag.retrieval.graph_retriever import GraphRetriever
        from minimed_rag.storage.neo4j import connect as neo4j_connect
        from minimed_rag.storage.postgres import connect as postgres_connect
        from minimed_rag.storage.repositories.term_repo import TermRepository

        settings = get_settings()
        neo4j = neo4j_connect(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password,
            settings.neo4j_database,
        )
        postgres = None
        fallback = None
        try:
            postgres = postgres_connect(settings.postgres_dsn)
            fallback = SynonymFallbackLinker(term_repo=TermRepository(postgres=postgres))
        except Exception as exc:
            logger.warning("Postgres unavailable, falling back to scispacy-only linker: %s", exc)

        scispacy_linker = ScispacyEntityLinker(
            model_name=settings.scispacy_model,
            use_umls_linker=settings.scispacy_umls_linker,
        )
        graph_client = GraphClient(neo4j=neo4j)
        question_linker = QuestionEntityLinker(
            scispacy_linker=scispacy_linker,
            fallback_linker=fallback,
            graph_client=graph_client,
        )

        def _factory():
            return GraphRetriever(
                graph_client=graph_client,
                min_confidence=kg_min_confidence,
                per_entity_limit=kg_per_entity_limit,
                max_paths=kg_max_paths,
                hops=kg_hops,
            )

        return _GraphComponents(
            graph_client=graph_client,
            question_linker=question_linker,
            graph_retriever_factory=_factory,
            neo4j=neo4j,
            postgres=postgres,
        )
    except Exception as exc:
        logger.warning("Failed to build graph components: %s", exc)
        return None


def _build_kg_augmented_retriever(
    *,
    text_retriever,
    graph_components: _GraphComponents,
    text_top_k: int | None,
    name: str,
):
    from minimed_rag.retrieval.kg_augmented_retriever import KGAugmentedRetriever

    return KGAugmentedRetriever(
        text_retriever=text_retriever,
        graph_retriever=graph_components.make_graph_retriever(),
        question_linker=graph_components.question_linker,
        text_top_k=text_top_k,
        name=name,
    )


def _load_retrieval_overrides(config_path: str) -> dict:
    from minimed_rag.common.config import load_yaml

    data = load_yaml(config_path)
    retrieval = data.get("retrieval", data)
    dense = retrieval.get("dense", {}) if isinstance(retrieval, dict) else {}
    bm25 = retrieval.get("bm25", {}) if isinstance(retrieval, dict) else {}
    reranker = retrieval.get("reranker", {}) if isinstance(retrieval, dict) else {}
    kg = retrieval.get("kg", {}) if isinstance(retrieval, dict) else {}

    overrides = {
        "retriever": retrieval.get("retriever") if isinstance(retrieval, dict) else None,
        "top_k": retrieval.get("top_k") if isinstance(retrieval, dict) else None,
        "max_context_chars": (
            retrieval.get("max_context_chars") if isinstance(retrieval, dict) else None
        ),
        "graph_budget_ratio": (
            retrieval.get("graph_budget_ratio") if isinstance(retrieval, dict) else None
        ),
        "index_path": bm25.get("index_path"),
        "dense_index_path": dense.get("index_path"),
        "embedding_model": dense.get("model_name"),
        "reranker_model": reranker.get("model_name"),
        "use_kg": retrieval.get("use_kg") if isinstance(retrieval, dict) else None,
        "kg_min_confidence": kg.get("min_confidence"),
        "kg_hops": kg.get("hops"),
        "kg_max_paths": kg.get("max_paths"),
        "kg_per_entity_limit": kg.get("per_entity_limit"),
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
