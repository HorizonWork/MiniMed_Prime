"""Phase 5 graph retriever.

Given a set of ``LinkedMention`` anchors from the question entity linker,
materialise 1-hop and (optionally) 2-hop ``GraphPath`` objects from the
reified-Assertion KG via ``GraphClient``. Paths are deduplicated, sorted
by weakest-link confidence, and capped at ``max_paths``.

Each ``GraphPath`` serialises to a single line of context that the
``ContextBuilder`` injects into the prompt under ``[Graph Evidence]``.

This module owns ``GraphFact`` — a single-hop unit shared with
``rag_pipeline.py`` — to keep the data model in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from minimed_rag.kg.graph_client import GraphClient, Neighbor, TwoHopPath


@dataclass(slots=True)
class GraphFact:
    """A single reified assertion ready for prompt rendering."""

    subject_key: str
    subject_name: str
    predicate: str
    object_key: str
    object_name: str
    confidence: float
    polarity: str
    source_system: str
    assertion_id: str


@dataclass(slots=True)
class GraphPath:
    """1- or 2-hop reified path between linked entities."""

    hops: list[GraphFact]
    anchor_mention: str
    metadata: dict = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        """Weakest-link confidence — drives ranking and pruning."""
        if not self.hops:
            return 0.0
        return min(hop.confidence for hop in self.hops)

    @property
    def length(self) -> int:
        return len(self.hops)

    @property
    def terminal_key(self) -> str:
        return self.hops[-1].object_key if self.hops else ""

    @property
    def predicate_chain(self) -> tuple[str, ...]:
        return tuple(hop.predicate for hop in self.hops)

    def serialize(self) -> str:
        """Render to a single human/LLM-readable line.

        1-hop: ``Metformin --TREATS--> T2D``
        2-hop: ``Metformin --TARGETS--> AMPK ; AMPK --GENE_ASSOCIATED_WITH_DISEASE--> T2D``
        """
        if not self.hops:
            return ""
        return " ; ".join(_serialize_fact(hop) for hop in self.hops)


def _serialize_fact(fact: GraphFact) -> str:
    predicate = fact.predicate.upper()
    if fact.polarity == "negative":
        predicate = f"NOT {predicate}"
    return f"{fact.subject_name} --{predicate}--> {fact.object_name}"


def _hop_from_neighbor(anchor_key: str, anchor_name: str, neighbor: Neighbor) -> GraphFact:
    """Convert a 1-hop ``Neighbor`` (oriented from the anchor) into a directed ``GraphFact``."""
    if neighbor.direction == "out":
        subj_key, obj_key = anchor_key, neighbor.node.key
        subj_name = anchor_name
        obj_name = neighbor.node.preferred_name or neighbor.node.key or "?"
    else:
        subj_key, obj_key = neighbor.node.key, anchor_key
        subj_name = neighbor.node.preferred_name or neighbor.node.key or "?"
        obj_name = anchor_name
    return GraphFact(
        subject_key=subj_key,
        subject_name=subj_name,
        predicate=neighbor.predicate,
        object_key=obj_key,
        object_name=obj_name,
        confidence=neighbor.confidence,
        polarity=neighbor.polarity,
        source_system=neighbor.source_system,
        assertion_id=neighbor.assertion_id,
    )


def _hops_from_two_hop(anchor_name: str, two_hop: TwoHopPath) -> list[GraphFact]:
    """Render a Cypher TwoHopPath as two directed GraphFacts.

    Edges keep the true KG SUBJECT -> OBJECT direction, even if the
    traversal discovered an assertion by walking into its object endpoint.
    """
    names = {
        two_hop.anchor.key: anchor_name or two_hop.anchor.preferred_name or two_hop.anchor.key or "?",
        two_hop.mid.key: two_hop.mid.preferred_name or two_hop.mid.key or "?",
        two_hop.terminal.key: two_hop.terminal.preferred_name or two_hop.terminal.key or "?",
    }
    return [
        GraphFact(
            subject_key=edge.subject_key,
            subject_name=names.get(edge.subject_key, edge.subject_key or "?"),
            predicate=edge.predicate,
            object_key=edge.object_key,
            object_name=names.get(edge.object_key, edge.object_key or "?"),
            confidence=edge.confidence,
            polarity=edge.polarity,
            source_system=edge.source_system,
            assertion_id=edge.assertion_id,
        )
        for edge in (two_hop.edge1, two_hop.edge2)
    ]


class GraphRetriever:
    """Walks the KG from linked-question anchors and returns ``GraphPath``s."""

    def __init__(
        self,
        graph_client: GraphClient,
        *,
        min_confidence: float = 0.5,
        per_entity_limit: int = 10,
        max_paths: int = 20,
        hops: int = 2,
    ) -> None:
        if hops not in (1, 2):
            raise ValueError(f"GraphRetriever supports hops in {{1, 2}}, got {hops}")
        self.graph_client = graph_client
        self.min_confidence = min_confidence
        self.per_entity_limit = per_entity_limit
        self.max_paths = max_paths
        self.hops = hops

    def retrieve(self, linked_mentions) -> list[GraphPath]:
        """Materialise paths for every linked mention with a CUI / entity_id."""
        seen: set[tuple] = set()
        paths: list[GraphPath] = []

        for mention in linked_mentions or []:
            anchor_key = (
                getattr(mention, "entity_id", None) or getattr(mention, "cui", None) or ""
            )
            if not anchor_key:
                continue
            anchor_name = getattr(mention, "mention_text", anchor_key) or anchor_key

            paths.extend(self._one_hop(anchor_key, anchor_name, mention, seen))
            if self.hops >= 2:
                paths.extend(self._two_hop(anchor_key, anchor_name, mention, seen))

        paths.sort(key=lambda p: p.confidence, reverse=True)
        return paths[: self.max_paths]

    def _one_hop(
        self,
        anchor_key: str,
        anchor_name: str,
        mention,
        seen: set[tuple],
    ) -> list[GraphPath]:
        try:
            neighbors = self.graph_client.get_neighbors(
                anchor_key,
                depth=1,
                limit=self.per_entity_limit,
                min_confidence=self.min_confidence,
            )
        except Exception:
            return []
        out: list[GraphPath] = []
        for neighbor in neighbors:
            hop = _hop_from_neighbor(anchor_key, anchor_name, neighbor)
            key = (1, hop.subject_key, hop.predicate, hop.object_key)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                GraphPath(
                    hops=[hop],
                    anchor_mention=getattr(mention, "mention_text", "") or anchor_name,
                )
            )
        return out

    def _two_hop(
        self,
        anchor_key: str,
        anchor_name: str,
        mention,
        seen: set[tuple],
    ) -> list[GraphPath]:
        try:
            two_hops = self.graph_client.get_two_hop_paths(
                anchor_key,
                limit=self.per_entity_limit,
                min_confidence=self.min_confidence,
            )
        except Exception:
            return []
        out: list[GraphPath] = []
        for raw_path in two_hops:
            hops = _hops_from_two_hop(anchor_name, raw_path)
            key = (
                2,
                hops[0].subject_key,
                hops[0].predicate,
                hops[1].predicate,
                hops[1].object_key,
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(
                GraphPath(
                    hops=hops,
                    anchor_mention=getattr(mention, "mention_text", "") or anchor_name,
                )
            )
        return out

    def serialize_paths(self, paths: list[GraphPath]) -> list[str]:
        return [p.serialize() for p in paths if p.hops]


__all__ = ["GraphFact", "GraphPath", "GraphRetriever"]
