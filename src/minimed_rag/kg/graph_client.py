"""Runtime KG query client.

Exposes the minimal read API the RAG pipeline needs:

- ``get_neighbors(key, depth=1, …)`` — one-hop neighbor expansion anchored
  on an Entity / Concept, filtered by per-Assertion confidence.
- ``get_path(cui_a, cui_b, max_depth=4)`` — shortest reified path between
  two CUIs.
- ``find_entity_by_cui`` / ``find_entity_by_name`` / ``find_concept_by_cui``
  — lightweight lookups used by the entity linker to bridge question
  mentions into the graph.

All Cypher assumes the Phase 4 reified-Assertion schema applied by
``kg_build/neo4j_loader.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from minimed_rag.kg.assertion_validator import AssertionDirectionValidator


@dataclass(frozen=True, slots=True)
class Node:
    labels: tuple[str, ...]
    entity_id: str | None = None
    cui: str | None = None
    preferred_name: str | None = None
    entity_type: str | None = None
    canonical_cui: str | None = None
    source_system: str | None = None
    source_id: str | None = None
    semantic_types: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return self.entity_id or self.cui or ""


@dataclass(frozen=True, slots=True)
class Edge:
    assertion_id: str
    predicate: str
    subject_key: str
    object_key: str
    confidence: float
    polarity: str
    source_system: str


@dataclass(frozen=True, slots=True)
class Neighbor:
    node: Node
    predicate: str
    confidence: float
    polarity: str
    source_system: str
    assertion_id: str
    direction: str


@dataclass(frozen=True, slots=True)
class TwoHopPath:
    """Two assertions touched while expanding anchor -> mid -> terminal.

    ``edge1`` and ``edge2`` keep the true SUBJECT -> OBJECT direction from the
    KG, which may differ from the traversal direction used to discover them.
    """

    anchor: Node
    mid: Node
    terminal: Node
    edge1: Edge
    edge2: Edge

    @property
    def confidence(self) -> float:
        """Weakest-link confidence — used for sorting/pruning."""
        return min(self.edge1.confidence, self.edge2.confidence)


GET_NEIGHBORS_DEPTH_1_CYPHER = """
MATCH (anchor)
WHERE (anchor:Entity AND (anchor.entity_id = $key OR anchor.canonical_cui = $key))
   OR (anchor:Concept AND anchor.cui = $key)
MATCH (a:Assertion)-[role_in:SUBJECT|OBJECT]->(anchor)
WHERE a.is_current = true AND a.confidence >= $min_conf
MATCH (a)-[role_out:SUBJECT|OBJECT]->(neighbor)
WHERE elementId(neighbor) <> elementId(anchor)
MATCH (a)-[:SUBJECT]->(subject)
MATCH (a)-[:OBJECT]->(object)
RETURN DISTINCT neighbor, a AS assertion, subject, object,
       type(role_in) AS role_in, type(role_out) AS role_out,
       labels(neighbor) AS neighbor_labels,
       labels(subject) AS subject_labels,
       labels(object) AS object_labels
ORDER BY a.confidence DESC
LIMIT $limit
""".strip()


GET_PATH_CYPHER_TEMPLATE = """
MATCH (a), (b)
WHERE ((a:Entity AND (a.entity_id = $cui_a OR a.canonical_cui = $cui_a))
    OR (a:Concept AND a.cui = $cui_a))
  AND ((b:Entity AND (b.entity_id = $cui_b OR b.canonical_cui = $cui_b))
    OR (b:Concept AND b.cui = $cui_b))
MATCH p = shortestPath((a)-[:SUBJECT|OBJECT*..__MAX_DEPTH__]-(b))
RETURN [n IN nodes(p) | {labels: labels(n), props: properties(n)}] AS nodes,
       [r IN relationships(p) | type(r)] AS rel_types,
       length(p) AS path_len
""".strip()


def _build_get_path_cypher(max_depth: int) -> str:
    return GET_PATH_CYPHER_TEMPLATE.replace("__MAX_DEPTH__", str(int(max_depth)))


GET_TWO_HOP_PATHS_CYPHER = """
MATCH (anchor)
WHERE (anchor:Entity AND (anchor.entity_id = $key OR anchor.canonical_cui = $key))
   OR (anchor:Concept AND anchor.cui = $key)
MATCH (a1:Assertion)-[role_anchor_1:SUBJECT|OBJECT]->(anchor)
MATCH (a1)-[role_mid_1:SUBJECT|OBJECT]->(mid)
MATCH (a1)-[:SUBJECT]->(subject1)
MATCH (a1)-[:OBJECT]->(object1)
WHERE a1.is_current = true
  AND a1.confidence >= $min_conf
  AND elementId(mid) <> elementId(anchor)
  AND type(role_anchor_1) <> type(role_mid_1)
MATCH (a2:Assertion)-[role_mid_2:SUBJECT|OBJECT]->(mid)
MATCH (a2)-[role_terminal_2:SUBJECT|OBJECT]->(terminal)
MATCH (a2)-[:SUBJECT]->(subject2)
MATCH (a2)-[:OBJECT]->(object2)
WHERE a2.is_current = true
  AND a2.confidence >= $min_conf
  AND elementId(a2) <> elementId(a1)
  AND elementId(terminal) <> elementId(anchor)
  AND elementId(terminal) <> elementId(mid)
  AND type(role_mid_2) <> type(role_terminal_2)
RETURN DISTINCT
       anchor, a1 AS edge1, mid, a2 AS edge2, terminal,
       subject1, object1, subject2, object2,
       labels(anchor) AS anchor_labels,
       labels(mid) AS mid_labels,
       labels(terminal) AS terminal_labels,
       labels(subject1) AS subject1_labels,
       labels(object1) AS object1_labels,
       labels(subject2) AS subject2_labels,
       labels(object2) AS object2_labels,
       type(role_anchor_1) AS role_anchor_1,
       type(role_mid_1) AS role_mid_1,
       type(role_mid_2) AS role_mid_2,
       type(role_terminal_2) AS role_terminal_2
ORDER BY (a1.confidence + a2.confidence) DESC
LIMIT $limit
""".strip()


FIND_ENTITY_BY_CUI_CYPHER = """
MATCH (e:Entity)
WHERE e.canonical_cui = $cui
RETURN e AS node, labels(e) AS node_labels
LIMIT $limit
""".strip()


FIND_ENTITY_BY_NAME_CYPHER = """
CALL db.index.fulltext.queryNodes('entity_name_ft', $name_query) YIELD node, score
WHERE node:Entity
RETURN node, labels(node) AS node_labels, score
ORDER BY score DESC
LIMIT $limit
""".strip()


FIND_CONCEPT_BY_CUI_CYPHER = """
MATCH (c:Concept {cui: $cui})
RETURN c AS node, labels(c) AS node_labels
LIMIT 1
""".strip()


def _node_from_row(raw: dict | None, labels: list[str] | None = None) -> Node | None:
    if raw is None:
        return None
    props = raw if isinstance(raw, dict) else dict(raw)
    label_tuple: tuple[str, ...] = tuple(labels) if labels else ()
    semantic = props.get("semantic_types") or ()
    if isinstance(semantic, list):
        semantic = tuple(semantic)
    return Node(
        labels=label_tuple,
        entity_id=props.get("entity_id"),
        cui=props.get("cui") or props.get("canonical_cui"),
        preferred_name=props.get("preferred_name"),
        entity_type=props.get("entity_type"),
        canonical_cui=props.get("canonical_cui"),
        source_system=props.get("source_system"),
        source_id=props.get("source_id"),
        semantic_types=semantic,  # type: ignore[arg-type]
    )


def _edge_from_assertion(
    assertion: dict,
    subject_key: str,
    object_key: str,
) -> Edge:
    return Edge(
        assertion_id=assertion.get("assertion_id", ""),
        predicate=assertion.get("predicate", ""),
        subject_key=subject_key,
        object_key=object_key,
        confidence=float(assertion.get("confidence", 0.0)),
        polarity=assertion.get("polarity", "positive"),
        source_system=assertion.get("source_system", ""),
    )


def _edge_from_roles(
    assertion: dict,
    left_key: str,
    left_role: str,
    right_key: str,
    right_role: str,
) -> Edge:
    if left_role == "SUBJECT" and right_role == "OBJECT":
        subject_key, object_key = left_key, right_key
    elif left_role == "OBJECT" and right_role == "SUBJECT":
        subject_key, object_key = right_key, left_key
    else:
        subject_key, object_key = left_key, right_key
    return _edge_from_assertion(assertion, subject_key, object_key)


def _assertion_matches_schema(
    assertion: dict,
    subject: Node | None,
    object_: Node | None,
    validator: AssertionDirectionValidator | None,
) -> bool:
    if validator is None:
        return True
    return validator.is_valid(
        assertion.get("predicate", ""),
        subject.entity_type if subject is not None else None,
        object_.entity_type if object_ is not None else None,
    )


@dataclass
class GraphClient:
    neo4j: Any
    assertion_validator: AssertionDirectionValidator | None = field(
        default_factory=AssertionDirectionValidator
    )

    def get_neighbors(
        self,
        key: str,
        depth: int = 1,
        limit: int = 50,
        min_confidence: float = 0.3,
    ) -> list[Neighbor]:
        """Return direct neighbors anchored on ``key``.

        ``key`` matches either ``Entity.entity_id``, ``Entity.canonical_cui``,
        or ``Concept.cui``. ``depth`` > 1 is a Phase 5 concern; raise so
        callers don't silently get depth-1 when expecting multi-hop.
        """
        if depth != 1:
            raise NotImplementedError(
                f"GraphClient.get_neighbors(depth={depth}) unsupported in Phase 4; "
                "only depth=1 is implemented."
            )
        rows = self.neo4j.query(
            GET_NEIGHBORS_DEPTH_1_CYPHER,
            {"key": key, "limit": limit, "min_conf": min_confidence},
        )
        results: list[Neighbor] = []
        for row in rows:
            node = _node_from_row(row.get("neighbor"), row.get("neighbor_labels"))
            assertion = row.get("assertion") or {}
            if isinstance(assertion, dict):
                assertion_data = assertion
            else:
                assertion_data = dict(assertion)
            subject = _node_from_row(row.get("subject"), row.get("subject_labels") or [])
            object_ = _node_from_row(row.get("object"), row.get("object_labels") or [])
            if not _assertion_matches_schema(
                assertion_data,
                subject,
                object_,
                self.assertion_validator,
            ):
                continue
            direction = "out" if row.get("role_in") == "SUBJECT" else "in"
            results.append(
                Neighbor(
                    node=node or Node(labels=()),
                    predicate=assertion_data.get("predicate", ""),
                    confidence=float(assertion_data.get("confidence", 0.0)),
                    polarity=assertion_data.get("polarity", "positive"),
                    source_system=assertion_data.get("source_system", ""),
                    assertion_id=assertion_data.get("assertion_id", ""),
                    direction=direction,
                )
            )
        return results

    def get_two_hop_paths(
        self,
        key: str,
        *,
        limit: int = 50,
        min_confidence: float = 0.5,
    ) -> list[TwoHopPath]:
        """Return 2-hop reified paths anchored on ``key``.

        One Cypher round-trip per call; the planner-side N+1 expansion is
        intentionally avoided here to keep latency bounded as the question
        entity count grows.
        """
        rows = self.neo4j.query(
            GET_TWO_HOP_PATHS_CYPHER,
            {"key": key, "limit": limit, "min_conf": min_confidence},
        )
        paths: list[TwoHopPath] = []
        for row in rows:
            anchor = _node_from_row(row.get("anchor"), row.get("anchor_labels") or [])
            mid = _node_from_row(row.get("mid"), row.get("mid_labels") or [])
            terminal = _node_from_row(row.get("terminal"), row.get("terminal_labels") or [])
            if anchor is None or mid is None or terminal is None:
                continue
            edge1_raw = row.get("edge1") or {}
            edge2_raw = row.get("edge2") or {}
            edge1_data = edge1_raw if isinstance(edge1_raw, dict) else dict(edge1_raw)
            edge2_data = edge2_raw if isinstance(edge2_raw, dict) else dict(edge2_raw)
            subject1 = _node_from_row(row.get("subject1"), row.get("subject1_labels") or [])
            object1 = _node_from_row(row.get("object1"), row.get("object1_labels") or [])
            subject2 = _node_from_row(row.get("subject2"), row.get("subject2_labels") or [])
            object2 = _node_from_row(row.get("object2"), row.get("object2_labels") or [])
            if not _assertion_matches_schema(
                edge1_data,
                subject1,
                object1,
                self.assertion_validator,
            ) or not _assertion_matches_schema(
                edge2_data,
                subject2,
                object2,
                self.assertion_validator,
            ):
                continue
            edge1 = _edge_from_roles(
                edge1_data,
                anchor.key,
                row.get("role_anchor_1", ""),
                mid.key,
                row.get("role_mid_1", ""),
            )
            edge2 = _edge_from_roles(
                edge2_data,
                mid.key,
                row.get("role_mid_2", ""),
                terminal.key,
                row.get("role_terminal_2", ""),
            )
            paths.append(
                TwoHopPath(
                    anchor=anchor,
                    mid=mid,
                    terminal=terminal,
                    edge1=edge1,
                    edge2=edge2,
                )
            )
        return paths

    def get_path(
        self,
        cui_a: str,
        cui_b: str,
        max_depth: int = 4,
    ) -> list[Node]:
        """Return nodes along the shortest path (empty list if no path)."""
        if not isinstance(max_depth, int) or max_depth < 1 or max_depth > 10:
            raise ValueError("max_depth must be an int in 1..10")
        cypher = _build_get_path_cypher(max_depth)
        rows = self.neo4j.query(cypher, {"cui_a": cui_a, "cui_b": cui_b})
        if not rows:
            return []
        row = rows[0]
        nodes_raw = row.get("nodes") or []
        path_nodes: list[Node] = []
        for item in nodes_raw:
            props = item.get("props") if isinstance(item, dict) else {}
            labels = item.get("labels") if isinstance(item, dict) else []
            n = _node_from_row(props, labels) if props is not None else None
            if n is not None:
                path_nodes.append(n)
        return path_nodes

    def find_entity_by_cui(self, cui: str, limit: int = 5) -> list[Node]:
        rows = self.neo4j.query(FIND_ENTITY_BY_CUI_CYPHER, {"cui": cui, "limit": limit})
        return [
            node
            for node in (_node_from_row(r.get("node"), r.get("node_labels")) for r in rows)
            if node is not None
        ]

    def find_entity_by_name(self, name: str, limit: int = 5) -> list[Node]:
        """Full-text match on Entity.preferred_name.

        Requires the ``entity_name_ft`` fulltext index (applied by
        ``Neo4jLoader.apply_schema``). Query is a Lucene string; callers
        should sanitize if taking raw user input.
        """
        rows = self.neo4j.query(
            FIND_ENTITY_BY_NAME_CYPHER,
            {"name_query": name, "limit": limit},
        )
        return [
            node
            for node in (_node_from_row(r.get("node"), r.get("node_labels")) for r in rows)
            if node is not None
        ]

    def find_concept_by_cui(self, cui: str) -> Node | None:
        rows = self.neo4j.query(FIND_CONCEPT_BY_CUI_CYPHER, {"cui": cui})
        if not rows:
            return None
        return _node_from_row(rows[0].get("node"), rows[0].get("node_labels"))
