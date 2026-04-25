"""Runtime knowledge-graph access layer.

``GraphClient`` is the single entry point RAG pipelines use to query the
Neo4j KG built in Phase 4. It's distinct from ``kg_build`` (the build-time
loader) because runtime queries are read-only, cached, and tolerant of
missing data in a way that bulk ingestion is not.
"""

from minimed_rag.kg.graph_client import Edge, GraphClient, Neighbor, Node

__all__ = ["Edge", "GraphClient", "Neighbor", "Node"]
