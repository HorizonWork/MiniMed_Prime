"""Generate reasoning paths CLI.

Single-question mode (Phase 6 acceptance):

    uv run minimed generate-paths \
        --question "What gene is mutated in cystic fibrosis?" \
        --output artifacts/paths/test.jsonl

Pipeline:
    QuestionEntityLinker -> TaskClassifier -> MetapathPlanner
        -> PathFinder (template-anchored or template-free)
        -> PathScorer -> PathPruner
        -> (optional) NegativeSampler
        -> JSONL writer (header + one ReasoningPath per line)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer

from minimed_rag.common.config import get_settings
from minimed_rag.domain.reasoning import ReasoningPath, ReasoningTask
from minimed_rag.kg.graph_client import GraphClient
from minimed_rag.nlp.linking.entity_linker import LinkedMention, QuestionEntityLinker
from minimed_rag.nlp.linking.scispacy_entity_linker import ScispacyEntityLinker
from minimed_rag.nlp.linking.synonym_fallback_linker import SynonymFallbackLinker
from minimed_rag.reasoning.negative_sampler import NegativeSampler
from minimed_rag.reasoning.path_finder import PathFinder
from minimed_rag.reasoning.path_pruner import PathPruner
from minimed_rag.reasoning.path_scorer import PathScorer
from minimed_rag.reasoning.task_classifier import TaskClassifier
from minimed_rag.schema_registry.metapath_registry import MetapathRegistry
from minimed_rag.schema_registry.predicate_registry import PredicateRegistry
from minimed_rag.storage.neo4j import connect as neo4j_connect
from minimed_rag.storage.postgres import connect as postgres_connect
from minimed_rag.storage.repositories.entity_stats_repo import EntityStatsRepo
from minimed_rag.storage.repositories.term_repo import TermRepository

logger = logging.getLogger(__name__)

app = typer.Typer(help="Generate reasoning paths")


class _GraphEntityTypeRepo:
    """Thin adapter exposing ``get_type(entity_id)`` for NegativeSampler."""

    def __init__(self, neo4j: Any) -> None:
        self.neo4j = neo4j
        self._cache: dict[str, str | None] = {}

    def get_type(self, entity_id: str) -> str | None:
        if entity_id in self._cache:
            return self._cache[entity_id]
        rows = self.neo4j.query(
            "MATCH (e:Entity {entity_id: $id}) RETURN e.entity_type AS entity_type LIMIT 1",
            {"id": entity_id},
        )
        result = rows[0].get("entity_type") if rows else None
        self._cache[entity_id] = result
        return result


def _task_id(question: str) -> str:
    return hashlib.sha1(question.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


def _serialize_mention(mention: LinkedMention) -> dict[str, Any]:
    return {
        "mention_text": mention.mention_text,
        "expanded_text": mention.expanded_text,
        "entity_id": mention.entity_id,
        "concept_id": mention.concept_id,
        "cui": mention.cui,
        "semantic_type": mention.semantic_type,
        "confidence": mention.confidence,
    }


def _write_jsonl(
    *,
    output: Path,
    question: str,
    task_type: str,
    graph_version: str,
    linked: list[LinkedMention],
    paths: list[ReasoningPath],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "_meta": {
            "question": question,
            "task_type": task_type,
            "graph_version": graph_version,
            "linked_entities": [_serialize_mention(m) for m in linked],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "num_paths": len(paths),
        }
    }
    with output.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(meta) + "\n")
        for path in paths:
            fh.write(json.dumps(asdict(path)) + "\n")


@app.callback(invoke_without_command=True)
def main(
    question: str = typer.Option(..., "--question", help="Biomedical question"),
    output: Path = typer.Option(..., "--output", help="JSONL output path"),
    graph_version: str | None = typer.Option(
        None, "--graph-version", help="Override graph_version (defaults from settings)"
    ),
    max_depth: int = typer.Option(3, "--max-depth", help="Maximum path length in hops"),
    limit: int = typer.Option(50, "--limit", help="Max paths to keep after pruning"),
    min_confidence: float = typer.Option(
        0.5, "--min-confidence", help="Per-edge confidence floor"
    ),
    include_negatives: bool = typer.Option(
        False,
        "--include-negatives/--no-include-negatives",
        help="Append negative paths (random walk + reversed + corrupted)",
    ),
    skip_scispacy_umls_linker: bool = typer.Option(
        False,
        "--skip-scispacy-umls-linker",
        help="Disable scispacy's bundled UMLS linker (term fallback only)",
    ),
) -> None:
    """Generate reasoning paths for ``--question`` and write them to ``--output``."""
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    gv = graph_version or settings.graph_version

    neo4j = neo4j_connect(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        settings.neo4j_database,
    )
    postgres = postgres_connect(settings.postgres_dsn)
    try:
        graph_client = GraphClient(neo4j=neo4j)
        term_repo = TermRepository(postgres=postgres)
        scispacy_linker = ScispacyEntityLinker(
            model_name=settings.scispacy_model,
            use_umls_linker=(settings.scispacy_umls_linker and not skip_scispacy_umls_linker),
        )
        fallback = SynonymFallbackLinker(term_repo=term_repo)
        question_linker = QuestionEntityLinker(
            scispacy_linker=scispacy_linker,
            fallback_linker=fallback,
            graph_client=graph_client,
        )

        linked = question_linker.link_question(question)

        classifier = TaskClassifier()
        task_type = classifier.classify(question)

        metapath_registry = MetapathRegistry()
        predicate_registry = PredicateRegistry()

        finder = PathFinder(
            neo4j=neo4j,
            metapath_registry=metapath_registry,
            predicate_registry=predicate_registry,
        )
        entity_stats_repo = EntityStatsRepo(postgres=postgres)
        scorer = PathScorer(
            metapath_registry=metapath_registry,
            entity_stats_repo=entity_stats_repo,
            hub_snapshot_id=gv,
        )
        pruner = PathPruner(min_score=0.05, top_k=limit)

        task_id = _task_id(question)
        candidates: list[ReasoningPath] = []
        for mention in linked:
            anchor_key = mention.entity_id or mention.cui
            if not anchor_key:
                continue
            if task_type == "general":
                candidates.extend(
                    finder.find_template_free_paths(
                        anchor_key,
                        task_id=task_id,
                        max_depth=max_depth,
                        min_confidence=min_confidence,
                        limit=100,
                        graph_version=gv,
                    )
                )
            else:
                candidates.extend(
                    finder.find_paths_from_anchor(
                        anchor_key,
                        task_type,
                        task_id=task_id,
                        max_depth=max_depth,
                        min_confidence=min_confidence,
                        limit_per_template=100,
                        graph_version=gv,
                    )
                )

        scored = scorer.score(candidates)
        deduped = pruner.dedupe(scored)
        kept = pruner.prune(deduped, limit=limit)

        if include_negatives:
            task = ReasoningTask(
                task_id=task_id,
                question=question,
                task_type=task_type,
                linked_question_entities=list(linked),
            )
            entity_repo = _GraphEntityTypeRepo(neo4j=neo4j)
            sampler = NegativeSampler(
                entity_linker=question_linker,
                path_finder=finder,
                predicate_registry=predicate_registry,
                entity_repo=entity_repo,
                entity_stats_repo=entity_stats_repo,
                hub_snapshot_id=gv,
            )
            kept.extend(sampler.generate(task, kept, gv))

        _write_jsonl(
            output=output,
            question=question,
            task_type=task_type,
            graph_version=gv,
            linked=linked,
            paths=kept,
        )

        typer.echo(f"Wrote {len(kept)} paths → {output}")
    finally:
        neo4j.close()
        if postgres is not None:
            postgres.close()
