"""Template-aware reasoning path finder over the reified Assertion KG.

For Phase 6 we need 1/2/3-hop paths from an anchor entity, optionally
constrained by a metapath template (predicate sequence + entity types
per hop). Templates may reference *inverse* predicate names (e.g.
``target_of`` is the inverse of ``targets``); we resolve to the canonical
predicate and flag those steps as ``template_inverse`` so the scorer can
treat them as legitimate forward traversals rather than "reversed"
penalties.

Anchor matching mirrors :data:`graph_client.GET_NEIGHBORS_DEPTH_1_CYPHER`
so the linker can hand us either an ``Entity.entity_id`` /
``Entity.canonical_cui`` or a ``Concept.cui`` and this module finds the
right starting node.
"""

from __future__ import annotations

import uuid
from typing import Any, Iterable

from minimed_rag.domain.reasoning import ReasoningPath, ReasoningPathStep
from minimed_rag.schema_registry.metapath_registry import (
    MetapathRegistry,
    MetapathStep,
    MetapathTemplate,
)
from minimed_rag.schema_registry.predicate_registry import PredicateRegistry


_ANCHOR_WHERE = (
    "((n0:Entity AND (n0.entity_id = $anchor_key OR n0.canonical_cui = $anchor_key)) "
    "OR (n0:Concept AND n0.cui = $anchor_key))"
)


_TEMPLATE_FREE_ONE_HOP = f"""
MATCH (n0)
WHERE {_ANCHOR_WHERE}
MATCH (a1:Assertion)-[r1a:SUBJECT|OBJECT]->(n0)
MATCH (a1)-[r1b:SUBJECT|OBJECT]->(n1)
WHERE a1.is_current = true
  AND a1.confidence >= $min_conf
  AND elementId(n1) <> elementId(n0)
RETURN n0, a1, n1,
       type(r1a) AS r1a, type(r1b) AS r1b,
       labels(n0) AS l0, labels(n1) AS l1
ORDER BY a1.confidence DESC
LIMIT $limit
""".strip()


_TEMPLATE_FREE_TWO_HOP = f"""
MATCH (n0)
WHERE {_ANCHOR_WHERE}
MATCH (a1:Assertion)-[r1a:SUBJECT|OBJECT]->(n0)
MATCH (a1)-[r1b:SUBJECT|OBJECT]->(n1)
WHERE a1.is_current = true
  AND a1.confidence >= $min_conf
  AND elementId(n1) <> elementId(n0)
MATCH (a2:Assertion)-[r2a:SUBJECT|OBJECT]->(n1)
MATCH (a2)-[r2b:SUBJECT|OBJECT]->(n2)
WHERE a2.is_current = true
  AND a2.confidence >= $min_conf
  AND elementId(a2) <> elementId(a1)
  AND elementId(n2) <> elementId(n0)
  AND elementId(n2) <> elementId(n1)
RETURN n0, a1, n1, a2, n2,
       type(r1a) AS r1a, type(r1b) AS r1b,
       type(r2a) AS r2a, type(r2b) AS r2b,
       labels(n0) AS l0, labels(n1) AS l1, labels(n2) AS l2
ORDER BY (a1.confidence + a2.confidence) DESC
LIMIT $limit
""".strip()


def _node_props(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return dict(raw)
    except Exception:  # noqa: BLE001
        return {}


def _node_key(props: dict[str, Any], labels: Iterable[str] | None) -> str:
    label_set = set(labels or ())
    if "Entity" in label_set:
        return props.get("entity_id") or props.get("canonical_cui") or props.get("cui") or ""
    if "Concept" in label_set:
        return props.get("cui") or ""
    return props.get("entity_id") or props.get("cui") or ""


class PathFinder:
    """Find reasoning paths anchored on a question entity through Neo4j."""

    def __init__(
        self,
        neo4j: Any,
        metapath_registry: MetapathRegistry,
        predicate_registry: PredicateRegistry,
    ) -> None:
        self.neo4j = neo4j
        self.metapath_registry = metapath_registry
        self.predicate_registry = predicate_registry

    # ---- public ----------------------------------------------------------

    def find_paths_from_anchor(
        self,
        anchor_key: str,
        task_type: str,
        *,
        task_id: str = "",
        max_depth: int = 3,
        min_confidence: float = 0.5,
        limit_per_template: int = 100,
        graph_version: str = "kg_local",
    ) -> list[ReasoningPath]:
        """Iterate templates for ``task_type`` and return all matching paths.

        Anchor-only mode: the answer entity is unknown; we expand from the
        anchor following each template's predicate / type sequence.
        """
        if not anchor_key:
            return []
        templates = [
            t for t in self.metapath_registry.get_for_task(task_type) if t.max_depth <= max_depth
        ]
        results: list[ReasoningPath] = []
        for template in templates:
            try:
                results.extend(
                    self._find_by_template_anchored(
                        anchor_key=anchor_key,
                        template=template,
                        min_confidence=min_confidence,
                        limit=limit_per_template,
                        task_id=task_id,
                        task_type=task_type,
                        graph_version=graph_version,
                    )
                )
            except Exception:  # noqa: BLE001
                # Robustness: bad Cypher / dirty data shouldn't crash the
                # whole run; skip this template and continue.
                continue
        return results

    def find_template_free_paths(
        self,
        anchor_key: str,
        *,
        task_id: str = "",
        max_depth: int = 2,
        min_confidence: float = 0.5,
        limit: int = 100,
        graph_version: str = "kg_local",
    ) -> list[ReasoningPath]:
        """Template-free anchor expansion (used when ``task_type='general'``)."""
        if not anchor_key:
            return []
        results: list[ReasoningPath] = []
        results.extend(
            self._template_free_n_hop(
                cypher=_TEMPLATE_FREE_ONE_HOP,
                hops=1,
                anchor_key=anchor_key,
                min_confidence=min_confidence,
                limit=limit,
                task_id=task_id,
                graph_version=graph_version,
            )
        )
        if max_depth >= 2:
            results.extend(
                self._template_free_n_hop(
                    cypher=_TEMPLATE_FREE_TWO_HOP,
                    hops=2,
                    anchor_key=anchor_key,
                    min_confidence=min_confidence,
                    limit=limit,
                    task_id=task_id,
                    graph_version=graph_version,
                )
            )
        return results

    def find_candidate_paths(
        self,
        question_entities: list,
        answer_entities: list,
        task_type: str,
        graph_version: str = "kg_local",
        *,
        min_confidence: float = 0.5,
        limit_per_template: int = 100,
        task_id: str = "",
    ) -> list[ReasoningPath]:
        """Phase 7 dataset mode: paths between question and answer entities.

        For Phase 6 we keep the implementation simple: run anchor-only paths
        from each question entity and post-filter to those terminating at an
        answer entity. Sufficient for negative sampling in single-question
        runs; Phase 7 may swap in pinned-endpoint Cypher for performance.
        """
        if not question_entities or not answer_entities:
            return []
        answer_keys = {self._entity_key(a) for a in answer_entities if self._entity_key(a)}
        if not answer_keys:
            return []
        results: list[ReasoningPath] = []
        for q_entity in question_entities:
            anchor_key = self._entity_key(q_entity)
            if not anchor_key:
                continue
            candidates = self.find_paths_from_anchor(
                anchor_key,
                task_type,
                task_id=task_id,
                min_confidence=min_confidence,
                limit_per_template=limit_per_template,
                graph_version=graph_version,
            )
            for path in candidates:
                if path.end_entity_id in answer_keys:
                    results.append(path)
        return results

    # ---- private ---------------------------------------------------------

    def _find_by_template_anchored(
        self,
        *,
        anchor_key: str,
        template: MetapathTemplate,
        min_confidence: float,
        limit: int,
        task_id: str,
        task_type: str,
        graph_version: str,
    ) -> list[ReasoningPath]:
        cypher, params = self._build_template_cypher(
            template=template,
            anchor_key=anchor_key,
            min_confidence=min_confidence,
            limit=limit,
        )
        rows = self.neo4j.query(cypher, params)
        results: list[ReasoningPath] = []
        for row in rows:
            path = self._row_to_template_path(
                row=row,
                template=template,
                task_id=task_id,
                task_type=task_type,
                graph_version=graph_version,
            )
            if path is not None:
                results.append(path)
        return results

    def _template_free_n_hop(
        self,
        *,
        cypher: str,
        hops: int,
        anchor_key: str,
        min_confidence: float,
        limit: int,
        task_id: str,
        graph_version: str,
    ) -> list[ReasoningPath]:
        rows = self.neo4j.query(
            cypher,
            {"anchor_key": anchor_key, "min_conf": min_confidence, "limit": limit},
        )
        results: list[ReasoningPath] = []
        for row in rows:
            path = self._row_to_template_free_path(
                row=row, hops=hops, task_id=task_id, graph_version=graph_version
            )
            if path is not None:
                results.append(path)
        return results

    def _build_template_cypher(
        self,
        *,
        template: MetapathTemplate,
        anchor_key: str,
        min_confidence: float,
        limit: int,
    ) -> tuple[str, dict[str, Any]]:
        """Compile a Cypher query for a given metapath template anchored on ``anchor_key``."""
        edge_floor = max(template.min_edge_confidence, min_confidence)
        params: dict[str, Any] = {"anchor_key": anchor_key, "limit": limit}

        match_clauses: list[str] = [f"MATCH (n0)\nWHERE {_ANCHOR_WHERE}"]
        where_extras: list[str] = []
        return_nodes: list[str] = ["n0"]
        return_edges: list[str] = []
        return_roles: list[str] = []
        return_labels: list[str] = ["labels(n0) AS l0"]
        order_by_terms: list[str] = []

        for idx, step in enumerate(template.pattern, start=1):
            canonical, _is_inverse, _ = self._resolve_predicate(
                step.predicate, step.subject_type, step.object_type
            )
            params[f"p{idx}"] = canonical
            params[f"min_conf{idx}"] = edge_floor

            prev_node = f"n{idx - 1}"
            this_node = f"n{idx}"
            assertion = f"a{idx}"
            ra = f"r{idx}a"
            rb = f"r{idx}b"

            match_clauses.append(
                f"MATCH ({assertion}:Assertion)-[{ra}:SUBJECT|OBJECT]->({prev_node})\n"
                f"MATCH ({assertion})-[{rb}:SUBJECT|OBJECT]->({this_node})"
            )

            step_where = [
                f"{assertion}.predicate = $p{idx}",
                f"{assertion}.is_current = true",
                f"{assertion}.confidence >= $min_conf{idx}",
                f"elementId({this_node}) <> elementId({prev_node})",
            ]
            for prior_idx in range(1, idx):
                step_where.append(f"elementId({assertion}) <> elementId(a{prior_idx})")
                step_where.append(f"elementId({this_node}) <> elementId(n{prior_idx - 1})")
                step_where.append(f"elementId({this_node}) <> elementId(n{prior_idx})")

            target_type = step.object_type
            if target_type:
                params[f"t{idx}"] = target_type
                step_where.append(
                    f"(({this_node}:Entity AND {this_node}.entity_type = $t{idx}) "
                    f"OR {this_node}:Concept)"
                )

            where_extras.append("\n  AND ".join(step_where))

            return_nodes.append(this_node)
            return_edges.append(f"{assertion}")
            return_roles.append(f"type({ra}) AS r{idx}a, type({rb}) AS r{idx}b")
            return_labels.append(f"labels({this_node}) AS l{idx}")
            order_by_terms.append(f"{assertion}.confidence")

        body = "\n".join(match_clauses)
        # First step's WHERE clause attaches to the first MATCH; subsequent steps' WHEREs
        # attach to that step's MATCH block. We bolt them all into one combined WHERE
        # for simplicity — Cypher allows that as long as identifiers are bound.
        combined_where = "\n  AND ".join(where_extras)
        if combined_where:
            body += f"\nWHERE {combined_where}"

        return_clause = ", ".join(
            return_nodes + return_edges + return_roles + return_labels
        )
        order_clause = " + ".join(order_by_terms) if order_by_terms else "1.0"

        cypher = (
            f"{body}\n"
            f"RETURN {return_clause}\n"
            f"ORDER BY {order_clause} DESC\n"
            f"LIMIT $limit"
        )
        return cypher, params

    def _row_to_template_path(
        self,
        *,
        row: dict[str, Any],
        template: MetapathTemplate,
        task_id: str,
        task_type: str,
        graph_version: str,
    ) -> ReasoningPath | None:
        path_id = uuid.uuid4().hex
        steps: list[ReasoningPathStep] = []

        for idx, template_step in enumerate(template.pattern, start=1):
            _canonical, is_inverse, is_directional = self._resolve_predicate(
                template_step.predicate,
                template_step.subject_type,
                template_step.object_type,
            )
            prev_props = _node_props(row.get(f"n{idx - 1}"))
            this_props = _node_props(row.get(f"n{idx}"))
            assertion_props = _node_props(row.get(f"a{idx}"))
            prev_labels = row.get(f"l{idx - 1}") or []
            this_labels = row.get(f"l{idx}") or []
            r_a = row.get(f"r{idx}a") or ""
            r_b = row.get(f"r{idx}b") or ""

            subject_key = _node_key(prev_props, prev_labels)
            object_key = _node_key(this_props, this_labels)
            if not subject_key or not object_key:
                return None

            direction = self._decode_direction(
                role_subj=r_a,
                role_obj=r_b,
                is_directional=is_directional,
                template_is_inverse=is_inverse,
            )

            steps.append(
                ReasoningPathStep(
                    path_id=path_id,
                    step_index=idx - 1,
                    subject_entity_id=subject_key,
                    predicate=template_step.predicate,
                    object_entity_id=object_key,
                    assertion_id=str(assertion_props.get("assertion_id", "")),
                    edge_confidence=float(assertion_props.get("confidence", 0.0)),
                    direction=direction,
                )
            )

        if not steps:
            return None
        return ReasoningPath(
            path_id=path_id,
            task_id=task_id,
            path_type="positive",
            start_entity_id=steps[0].subject_entity_id,
            end_entity_id=steps[-1].object_entity_id,
            task_type=task_type,
            metapath_template_id=template.name,
            graph_version=graph_version,
            steps=steps,
        )

    def _row_to_template_free_path(
        self,
        *,
        row: dict[str, Any],
        hops: int,
        task_id: str,
        graph_version: str,
    ) -> ReasoningPath | None:
        path_id = uuid.uuid4().hex
        steps: list[ReasoningPathStep] = []
        for idx in range(1, hops + 1):
            prev_props = _node_props(row.get(f"n{idx - 1}"))
            this_props = _node_props(row.get(f"n{idx}"))
            assertion_props = _node_props(row.get(f"a{idx}"))
            prev_labels = row.get(f"l{idx - 1}") or []
            this_labels = row.get(f"l{idx}") or []
            r_a = row.get(f"r{idx}a") or ""
            r_b = row.get(f"r{idx}b") or ""

            subject_key = _node_key(prev_props, prev_labels)
            object_key = _node_key(this_props, this_labels)
            if not subject_key or not object_key:
                return None

            predicate = str(assertion_props.get("predicate", ""))
            # Template-free traversal: we don't know template subject/object types,
            # so we resolve by name only. is_inverse stays False — direction is
            # decoded purely from the row's SUBJECT/OBJECT roles.
            _canonical, is_inverse, is_directional = self._resolve_predicate(
                predicate, "", ""
            )
            direction = self._decode_direction(
                role_subj=r_a,
                role_obj=r_b,
                is_directional=is_directional,
                template_is_inverse=is_inverse,
            )

            steps.append(
                ReasoningPathStep(
                    path_id=path_id,
                    step_index=idx - 1,
                    subject_entity_id=subject_key,
                    predicate=predicate,
                    object_entity_id=object_key,
                    assertion_id=str(assertion_props.get("assertion_id", "")),
                    edge_confidence=float(assertion_props.get("confidence", 0.0)),
                    direction=direction,
                )
            )
        if not steps:
            return None
        return ReasoningPath(
            path_id=path_id,
            task_id=task_id,
            path_type="positive",
            start_entity_id=steps[0].subject_entity_id,
            end_entity_id=steps[-1].object_entity_id,
            task_type="general",
            metapath_template_id="",
            graph_version=graph_version,
            steps=steps,
        )

    def _resolve_predicate(
        self, name: str, subject_type: str, object_type: str
    ) -> tuple[str, bool, bool]:
        """Resolve a template predicate to ``(canonical, traversal_is_inverse, is_directional)``.

        ``traversal_is_inverse`` means the template traverses against the
        canonical predicate's stored direction:

        - ``name`` is a primary predicate AND template ``(subject, object)``
          types match ``(domain, range)`` → forward.
        - ``name`` is a primary predicate AND template types match
          ``(range, domain)`` → inverse traversal of the same canonical
          predicate (e.g. ``[Disease, involved_in, Pathway]`` while
          ``involved_in: Pathway → Disease``).
        - ``name`` is the inverse of some primary → query the primary,
          inverse traversal.
        - Unknown predicate → ``(name, False, True)`` (treat as directional,
          query as-is).
        """
        try:
            rule = self.predicate_registry.get(name)
        except KeyError:
            for primary_name, primary_rule in self.predicate_registry.all().items():
                if primary_rule.inverse == name:
                    return primary_name, True, primary_rule.directional
            return name, False, True

        if subject_type or object_type:
            forward = self.predicate_registry.matches_type(
                subject_type, rule.domain
            ) and self.predicate_registry.matches_type(object_type, rule.range)
            inverse = self.predicate_registry.matches_type(
                subject_type, rule.range
            ) and self.predicate_registry.matches_type(object_type, rule.domain)
            if forward and not inverse:
                return name, False, rule.directional
            if inverse and not forward:
                return name, True, rule.directional
        return name, False, rule.directional

    @staticmethod
    def _decode_direction(
        *,
        role_subj: str,
        role_obj: str,
        is_directional: bool,
        template_is_inverse: bool,
    ) -> str:
        """Map row roles + template intent to a step ``direction`` tag.

        - non-directional predicate → ``forward`` (commutative, no penalty).
        - template_is_inverse + roles swapped (subj=OBJECT, obj=SUBJECT) →
          ``template_inverse`` (legitimate; no penalty in the scorer).
        - directional, forward template, roles match (subj=SUBJECT, obj=OBJECT) →
          ``forward``.
        - anything else (data direction unexpectedly opposite of template) →
          ``reversed``.
        """
        if not is_directional:
            return "forward"
        if template_is_inverse:
            if role_subj == "OBJECT" and role_obj == "SUBJECT":
                return "template_inverse"
            return "reversed"
        if role_subj == "SUBJECT" and role_obj == "OBJECT":
            return "forward"
        return "reversed"

    @staticmethod
    def _entity_key(entity: Any) -> str:
        if isinstance(entity, str):
            return entity
        return (
            getattr(entity, "entity_id", None)
            or getattr(entity, "cui", None)
            or getattr(entity, "concept_id", None)
            or ""
        )
