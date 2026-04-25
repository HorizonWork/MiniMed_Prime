"""Negative path sampler for TRM training (Phase 7 input).

Generates invalid reasoning paths from positive paths using six
strategies in Phase 6:

- ``wrong_answer_paths``: paths that resolve to a wrong MC option
  (only when ``task.options`` is non-empty).
- ``reversed_direction_paths``: directional positive paths flipped
  end-to-end.
- ``relation_corruption_paths``: positive paths with one predicate
  swapped for an incompatible one.
- ``random_walk_paths``: portable Cypher random-walk anchored on the
  question entity.
- ``generic_hub_paths``: 2-hop paths anchored on the question entity
  that pass through a top-1%-degree hub (gated on ``entity_stats_repo``).
- ``unsupported_paths``: paths through Assertion nodes flagged with
  elevated ``conflict_score`` by the offline ``ConflictDetector``.

The single remaining stub is ``entity_corruption_paths``, which still
needs SapBERT-style semantic similarity (foundation backlog item #9).
"""

from __future__ import annotations

import copy
import random
import uuid
from typing import Any

from minimed_rag.domain.enums import PathType, ValidityLabel
from minimed_rag.domain.reasoning import ReasoningPath, ReasoningPathStep


class NegativeSampler:
    def __init__(
        self,
        entity_linker: Any,
        path_finder: Any,
        predicate_registry: Any,
        entity_repo: Any,
        *,
        num_random_walks: int = 5,
        random_walk_length: int = 2,
        rng: random.Random | None = None,
        entity_stats_repo: Any | None = None,
        hub_snapshot_id: str = "",
        hub_top_pct: float = 0.01,
        conflict_threshold: float = 0.5,
    ) -> None:
        self.entity_linker = entity_linker
        self.path_finder = path_finder
        self.predicate_registry = predicate_registry
        self.entity_repo = entity_repo
        self.num_random_walks = num_random_walks
        self.random_walk_length = random_walk_length
        self.rng = rng or random.Random()
        self.entity_stats_repo = entity_stats_repo
        self.hub_snapshot_id = hub_snapshot_id
        self.hub_top_pct = hub_top_pct
        self.conflict_threshold = conflict_threshold
        self._hub_ids_cached: list[str] | None = None

    def generate(
        self,
        task,
        positive_paths: list[ReasoningPath],
        graph_version: str,
    ) -> list[ReasoningPath]:
        negatives: list[ReasoningPath] = []
        if getattr(task, "options", None):
            negatives += self.generate_wrong_answer_paths(task, graph_version)
        negatives += self.generate_reversed_direction_paths(positive_paths)
        negatives += self.generate_relation_corruption_paths(positive_paths)
        negatives += self.generate_random_walk_paths(task, graph_version)
        negatives += self.generate_generic_hub_paths(task, graph_version)
        negatives += self.generate_entity_corruption_paths(positive_paths)
        negatives += self.generate_unsupported_paths(task, graph_version)
        return self.balance_negatives(negatives, max_per_task=10)

    def generate_wrong_answer_paths(self, task, graph_version: str) -> list[ReasoningPath]:
        wrong_answers = [
            option
            for option in (task.options or [])
            if getattr(option, "text", None) and option.text != task.correct_answer
        ]
        paths: list[ReasoningPath] = []
        for wrong in wrong_answers:
            linked_wrong = self.entity_linker.link_answer(wrong)
            if not linked_wrong:
                continue
            candidates = self.path_finder.find_candidate_paths(
                task.linked_question_entities,
                linked_wrong,
                task.task_type,
                graph_version,
            )
            for candidate in candidates:
                negative = copy.deepcopy(candidate)
                negative.path_id = uuid.uuid4().hex
                negative.path_type = PathType.HARD_NEGATIVE.value
                negative.validity_label = ValidityLabel.INVALID.value
                negative.negative_type = "wrong_answer_path"
                paths.append(negative)
        return paths

    def generate_reversed_direction_paths(
        self, positive_paths: list[ReasoningPath]
    ) -> list[ReasoningPath]:
        negatives: list[ReasoningPath] = []
        for path in positive_paths:
            reversed_steps: list[ReasoningPathStep] = []
            for step in reversed(path.steps):
                if not self._is_directional(step.predicate):
                    continue
                flipped = copy.copy(step)
                flipped.subject_entity_id = step.object_entity_id
                flipped.object_entity_id = step.subject_entity_id
                flipped.direction = "reversed"
                reversed_steps.append(flipped)
            if not reversed_steps:
                continue
            for idx, step in enumerate(reversed_steps):
                step.step_index = idx
            negative = copy.copy(path)
            negative.path_id = uuid.uuid4().hex
            negative.path_type = PathType.HARD_NEGATIVE.value
            negative.validity_label = ValidityLabel.INVALID.value
            negative.negative_type = "reversed_causal_direction"
            negative.steps = reversed_steps
            negative.start_entity_id = reversed_steps[0].subject_entity_id
            negative.end_entity_id = reversed_steps[-1].object_entity_id
            negatives.append(negative)
        return negatives

    def generate_relation_corruption_paths(
        self, positive_paths: list[ReasoningPath]
    ) -> list[ReasoningPath]:
        negatives: list[ReasoningPath] = []
        for path in positive_paths:
            if not path.steps:
                continue
            corrupted = copy.deepcopy(path)
            corrupted.path_id = uuid.uuid4().hex
            step = self.rng.choice(corrupted.steps)
            subj_type = self._get_entity_type(step.subject_entity_id) or ""
            obj_type = self._get_entity_type(step.object_entity_id) or ""
            try:
                step.predicate = self.predicate_registry.sample_incompatible_predicate(
                    subj_type, obj_type, step.predicate
                )
            except Exception:  # noqa: BLE001
                continue
            corrupted.path_type = PathType.HARD_NEGATIVE.value
            corrupted.validity_label = ValidityLabel.INVALID.value
            corrupted.negative_type = "relation_type_mismatch"
            negatives.append(corrupted)
        return negatives

    def generate_random_walk_paths(self, task, graph_version: str) -> list[ReasoningPath]:
        """Anchor-rooted random walks using portable Cypher (no APOC).

        For each linked question entity, run ``num_random_walks`` walks of
        length ``random_walk_length`` and tag the result as
        ``random_negative``.
        """
        negatives: list[ReasoningPath] = []
        for mention in getattr(task, "linked_question_entities", []) or []:
            anchor_key = self._mention_key(mention)
            if not anchor_key:
                continue
            for _ in range(self.num_random_walks):
                walk = self._random_walk(anchor_key, self.random_walk_length, graph_version)
                if walk is None:
                    continue
                walk.task_id = getattr(task, "task_id", "") or walk.task_id
                walk.task_type = getattr(task, "task_type", "") or walk.task_type
                negatives.append(walk)
        return negatives

    def generate_generic_hub_paths(self, task, graph_version: str) -> list[ReasoningPath]:
        """2-hop paths anchored on the question entity that pass through a
        top-degree hub. Tag as ``negative_type='generic_hub'``.

        Gated on ``entity_stats_repo`` + ``hub_snapshot_id``: returns ``[]``
        if either is missing or the snapshot has no entries — this lets
        the sampler degrade gracefully when graph_stats hasn't been
        computed yet.
        """
        if self.entity_stats_repo is None or not self.hub_snapshot_id:
            return []
        hub_ids = self._top_hub_entity_ids()
        if not hub_ids:
            return []
        negatives: list[ReasoningPath] = []
        for mention in getattr(task, "linked_question_entities", []) or []:
            anchor_key = self._mention_key(mention)
            if not anchor_key:
                continue
            rows = self._fetch_hub_paths(anchor_key, hub_ids, graph_version)
            for row in rows:
                path = self._row_to_hub_path(
                    row=row,
                    anchor_key=anchor_key,
                    task=task,
                    graph_version=graph_version,
                )
                if path is not None:
                    negatives.append(path)
        return negatives

    def generate_entity_corruption_paths(
        self, positive_paths: list[ReasoningPath]
    ) -> list[ReasoningPath]:
        # Deferred to Phase 7: entity corruption needs SapBERT-style
        # semantic similarity to pick a "plausibly-confusing" wrong entity.
        # Foundation backlog item #9.
        return []

    def generate_unsupported_paths(self, task, graph_version: str) -> list[ReasoningPath]:
        """Paths whose terminal Assertion has ``conflict_score`` above
        threshold (set by the offline :class:`ConflictDetector`).

        These are "unsupported by the KG itself" — the KG carries
        contradictory predicates on the same (subject, object), so any
        path terminating at the conflicted hop is unreliable. Tag as
        ``negative_type='unsupported_kg_conflict'`` and stamp
        ``max_conflict_score`` on the path.

        Returns ``[]`` if the live KG has no flagged conflicts (typical
        when ``ConflictDetector.run()`` hasn't been wired into the build
        pipeline yet — see negative_sampler docstring + cleanup notes).
        """
        negatives: list[ReasoningPath] = []
        for mention in getattr(task, "linked_question_entities", []) or []:
            anchor_key = self._mention_key(mention)
            if not anchor_key:
                continue
            rows = self._fetch_unsupported_paths(anchor_key)
            for row in rows:
                path = self._row_to_unsupported_path(
                    row=row,
                    anchor_key=anchor_key,
                    task=task,
                    graph_version=graph_version,
                )
                if path is not None:
                    negatives.append(path)
        return negatives

    def balance_negatives(
        self, negatives: list[ReasoningPath], max_per_task: int
    ) -> list[ReasoningPath]:
        if max_per_task is None or max_per_task >= len(negatives):
            return list(negatives)
        return list(negatives[:max_per_task])

    # ---- helpers ----------------------------------------------------------

    def _random_walk(
        self, anchor_key: str, length: int, graph_version: str
    ) -> ReasoningPath | None:
        """Walk ``length`` random hops from ``anchor_key`` and return as a path."""
        path_id = uuid.uuid4().hex
        steps: list[ReasoningPathStep] = []
        current_key = anchor_key
        for idx in range(length):
            neighbors = self._fetch_random_neighbors(current_key)
            if not neighbors:
                break
            choice = self.rng.choice(neighbors)
            next_key = choice.get("neighbor_key", "")
            assertion_id = choice.get("assertion_id", "")
            predicate = choice.get("predicate", "")
            confidence = float(choice.get("confidence", 0.0))
            if not next_key or next_key == current_key:
                break
            steps.append(
                ReasoningPathStep(
                    path_id=path_id,
                    step_index=idx,
                    subject_entity_id=current_key,
                    predicate=predicate,
                    object_entity_id=next_key,
                    assertion_id=assertion_id,
                    edge_confidence=confidence,
                    direction="forward",
                )
            )
            current_key = next_key
        if not steps:
            return None
        return ReasoningPath(
            path_id=path_id,
            task_id="",
            path_type=PathType.RANDOM_NEGATIVE.value,
            start_entity_id=anchor_key,
            end_entity_id=current_key,
            task_type="",
            metapath_template_id="",
            path_confidence=0.0,
            validity_label=ValidityLabel.INVALID.value,
            graph_version=graph_version,
            steps=steps,
            negative_type="random_walk",
        )

    def _fetch_random_neighbors(self, anchor_key: str) -> list[dict[str, Any]]:
        cypher = """
        MATCH (anchor)
        WHERE (anchor:Entity AND (anchor.entity_id = $anchor_key
                                   OR anchor.canonical_cui = $anchor_key))
           OR (anchor:Concept AND anchor.cui = $anchor_key)
        MATCH (a:Assertion)-[:SUBJECT|OBJECT]->(anchor)
        WHERE a.is_current = true
        MATCH (a)-[:SUBJECT|OBJECT]->(neighbor)
        WHERE elementId(neighbor) <> elementId(anchor)
        WITH anchor, a, neighbor, rand() AS r
        ORDER BY r
        LIMIT 25
        RETURN
          a.assertion_id AS assertion_id,
          a.predicate AS predicate,
          a.confidence AS confidence,
          coalesce(neighbor.entity_id, neighbor.canonical_cui, neighbor.cui) AS neighbor_key
        """
        try:
            return self.path_finder.neo4j.query(cypher, {"anchor_key": anchor_key})
        except Exception:  # noqa: BLE001
            return []

    def _is_directional(self, predicate: str) -> bool:
        try:
            return self.predicate_registry.is_directional(predicate)
        except KeyError:
            return True

    def _get_entity_type(self, entity_id: str) -> str | None:
        if self.entity_repo is None:
            return None
        try:
            return self.entity_repo.get_type(entity_id)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _mention_key(mention: Any) -> str:
        return (
            getattr(mention, "entity_id", None)
            or getattr(mention, "cui", None)
            or getattr(mention, "concept_id", None)
            or ""
        )

    # ---- generic_hub helpers ---------------------------------------------

    def _top_hub_entity_ids(self) -> list[str]:
        if self._hub_ids_cached is None:
            try:
                self._hub_ids_cached = self.entity_stats_repo.get_top_degree_entities(
                    self.hub_snapshot_id, top_pct=self.hub_top_pct
                )
            except Exception:  # noqa: BLE001
                self._hub_ids_cached = []
        return self._hub_ids_cached

    def _fetch_hub_paths(
        self, anchor_key: str, hub_ids: list[str], graph_version: str
    ) -> list[dict[str, Any]]:
        cypher = """
        MATCH (anchor)
        WHERE (anchor:Entity AND (anchor.entity_id = $anchor_key
                                   OR anchor.canonical_cui = $anchor_key))
           OR (anchor:Concept AND anchor.cui = $anchor_key)
        MATCH (a1:Assertion)-[:SUBJECT|OBJECT]->(anchor)
        MATCH (a1)-[:SUBJECT|OBJECT]->(hub:Entity)
        WHERE a1.is_current = true
          AND elementId(hub) <> elementId(anchor)
          AND hub.entity_id IN $hub_ids
        MATCH (a2:Assertion)-[:SUBJECT|OBJECT]->(hub)
        MATCH (a2)-[:SUBJECT|OBJECT]->(terminal)
        WHERE a2.is_current = true
          AND elementId(a2) <> elementId(a1)
          AND elementId(terminal) <> elementId(hub)
          AND elementId(terminal) <> elementId(anchor)
        WITH a1, hub, a2, terminal, rand() AS r
        ORDER BY r
        LIMIT 25
        RETURN
          a1.assertion_id AS a1_id, a1.predicate AS a1_pred, a1.confidence AS a1_conf,
          hub.entity_id   AS hub_key,
          a2.assertion_id AS a2_id, a2.predicate AS a2_pred, a2.confidence AS a2_conf,
          coalesce(terminal.entity_id, terminal.canonical_cui, terminal.cui) AS terminal_key
        """
        try:
            return self.path_finder.neo4j.query(
                cypher, {"anchor_key": anchor_key, "hub_ids": hub_ids}
            )
        except Exception:  # noqa: BLE001
            return []

    def _row_to_hub_path(
        self,
        *,
        row: dict[str, Any],
        anchor_key: str,
        task: Any,
        graph_version: str,
    ) -> ReasoningPath | None:
        hub_key = row.get("hub_key") or ""
        terminal_key = row.get("terminal_key") or ""
        if not hub_key or not terminal_key:
            return None
        path_id = uuid.uuid4().hex
        steps = [
            ReasoningPathStep(
                path_id=path_id,
                step_index=0,
                subject_entity_id=anchor_key,
                predicate=str(row.get("a1_pred") or ""),
                object_entity_id=hub_key,
                assertion_id=str(row.get("a1_id") or ""),
                edge_confidence=float(row.get("a1_conf") or 0.0),
                direction="forward",
            ),
            ReasoningPathStep(
                path_id=path_id,
                step_index=1,
                subject_entity_id=hub_key,
                predicate=str(row.get("a2_pred") or ""),
                object_entity_id=terminal_key,
                assertion_id=str(row.get("a2_id") or ""),
                edge_confidence=float(row.get("a2_conf") or 0.0),
                direction="forward",
            ),
        ]
        return ReasoningPath(
            path_id=path_id,
            task_id=getattr(task, "task_id", "") or "",
            path_type=PathType.RANDOM_NEGATIVE.value,
            start_entity_id=anchor_key,
            end_entity_id=terminal_key,
            task_type=getattr(task, "task_type", "") or "",
            metapath_template_id="",
            path_confidence=0.0,
            validity_label=ValidityLabel.INVALID.value,
            graph_version=graph_version,
            steps=steps,
            negative_type="generic_hub",
        )

    # ---- unsupported helpers ---------------------------------------------

    def _fetch_unsupported_paths(self, anchor_key: str) -> list[dict[str, Any]]:
        """1- and 2-hop paths whose **terminal** assertion has elevated
        ``conflict_score``. Intermediate-hop conflicts are excluded — the
        signal is cleanest when the conflict sits at the endpoint of the
        reasoning chain (the place that drives the answer)."""
        cypher = """
        MATCH (anchor)
        WHERE (anchor:Entity AND (anchor.entity_id = $anchor_key
                                   OR anchor.canonical_cui = $anchor_key))
           OR (anchor:Concept AND anchor.cui = $anchor_key)

        // 1-hop: terminal assertion is the conflicted one.
        OPTIONAL MATCH (a1c:Assertion)-[:SUBJECT|OBJECT]->(anchor)
        OPTIONAL MATCH (a1c)-[:SUBJECT|OBJECT]->(t1)
        WITH anchor,
             collect({hops: 1,
                      a1_id:   a1c.assertion_id,   a1_pred:  a1c.predicate,
                      a1_conf: a1c.confidence,     a1_score: a1c.conflict_score,
                      hub_key: null,
                      a2_id:   null,                a2_pred:  null,
                      a2_conf: null,                a2_score: null,
                      terminal_key: coalesce(t1.entity_id, t1.canonical_cui, t1.cui)})
               AS one_hops

        // 2-hop: terminal assertion (a2) is the conflicted one.
        OPTIONAL MATCH (a1:Assertion)-[:SUBJECT|OBJECT]->(anchor)
        OPTIONAL MATCH (a1)-[:SUBJECT|OBJECT]->(mid)
        OPTIONAL MATCH (a2:Assertion)-[:SUBJECT|OBJECT]->(mid)
        OPTIONAL MATCH (a2)-[:SUBJECT|OBJECT]->(t2)
        WITH one_hops,
             collect({hops: 2,
                      a1_id:   a1.assertion_id,   a1_pred:  a1.predicate,
                      a1_conf: a1.confidence,     a1_score: a1.conflict_score,
                      hub_key: coalesce(mid.entity_id, mid.canonical_cui, mid.cui),
                      a2_id:   a2.assertion_id,   a2_pred:  a2.predicate,
                      a2_conf: a2.confidence,     a2_score: a2.conflict_score,
                      terminal_key: coalesce(t2.entity_id, t2.canonical_cui, t2.cui)})
               AS two_hops

        UNWIND (one_hops + two_hops) AS row
        WITH row
        WHERE row.terminal_key IS NOT NULL
          AND ((row.hops = 1 AND row.a1_score >= $threshold)
            OR (row.hops = 2 AND row.a2_score >= $threshold))
        RETURN row
        LIMIT 25
        """
        try:
            return self.path_finder.neo4j.query(
                cypher,
                {"anchor_key": anchor_key, "threshold": self.conflict_threshold},
            )
        except Exception:  # noqa: BLE001
            return []

    def _row_to_unsupported_path(
        self,
        *,
        row: dict[str, Any],
        anchor_key: str,
        task: Any,
        graph_version: str,
    ) -> ReasoningPath | None:
        # Some Neo4j drivers wrap the inner map under "row"; some inline it.
        body = row.get("row") if "row" in row else row
        if not isinstance(body, dict):
            return None
        terminal_key = body.get("terminal_key") or ""
        if not terminal_key:
            return None
        hops = int(body.get("hops") or 0)
        path_id = uuid.uuid4().hex
        steps: list[ReasoningPathStep] = []

        if hops == 1:
            scores = [float(body.get("a1_score") or 0.0)]
            steps.append(
                ReasoningPathStep(
                    path_id=path_id,
                    step_index=0,
                    subject_entity_id=anchor_key,
                    predicate=str(body.get("a1_pred") or ""),
                    object_entity_id=terminal_key,
                    assertion_id=str(body.get("a1_id") or ""),
                    edge_confidence=float(body.get("a1_conf") or 0.0),
                    direction="forward",
                )
            )
        elif hops == 2:
            hub_key = body.get("hub_key") or ""
            if not hub_key:
                return None
            scores = [
                float(body.get("a1_score") or 0.0),
                float(body.get("a2_score") or 0.0),
            ]
            steps.extend(
                [
                    ReasoningPathStep(
                        path_id=path_id,
                        step_index=0,
                        subject_entity_id=anchor_key,
                        predicate=str(body.get("a1_pred") or ""),
                        object_entity_id=hub_key,
                        assertion_id=str(body.get("a1_id") or ""),
                        edge_confidence=float(body.get("a1_conf") or 0.0),
                        direction="forward",
                    ),
                    ReasoningPathStep(
                        path_id=path_id,
                        step_index=1,
                        subject_entity_id=hub_key,
                        predicate=str(body.get("a2_pred") or ""),
                        object_entity_id=terminal_key,
                        assertion_id=str(body.get("a2_id") or ""),
                        edge_confidence=float(body.get("a2_conf") or 0.0),
                        direction="forward",
                    ),
                ]
            )
        else:
            return None

        return ReasoningPath(
            path_id=path_id,
            task_id=getattr(task, "task_id", "") or "",
            path_type=PathType.HARD_NEGATIVE.value,
            start_entity_id=anchor_key,
            end_entity_id=terminal_key,
            task_type=getattr(task, "task_type", "") or "",
            metapath_template_id="",
            path_confidence=0.0,
            validity_label=ValidityLabel.INVALID.value,
            graph_version=graph_version,
            steps=steps,
            negative_type="unsupported_kg_conflict",
            max_conflict_score=max(scores) if scores else 0.0,
        )
