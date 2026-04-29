"""Pydantic v2 models for the PrimeKG-X schema (config/kg_schema.yaml).

Round-trip: `KGSchema.model_validate(yaml.safe_load(open("config/kg_schema.yaml")))`
must succeed. Validators check that every edge's head/tail names resolve to a
declared node type (or the literal "any" for cross-vocab XREF edges) and that
the meta counts match the actual collection sizes.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

ANY_NODE = "any"
REQUIRED_GLOBAL_EDGE_PROPS = {
    "source",
    "evidence_level",
    "n_pubmed_citations",
    "last_curated_date",
    "weight",
}


class NodeType(BaseModel):
    ids: list[str] = Field(min_length=1)
    props: list[str] = Field(default_factory=list)
    subtypes: list[str] | None = None


class EdgeType(BaseModel):
    head_type: str
    tail_type: str
    symmetric: bool = False
    dag: bool = False
    props: list[str] = Field(default_factory=list)


class SchemaMeta(BaseModel):
    total_node_types: int
    total_edge_types: int
    primekg_inherited: int
    clinical_extension: int


class KGSchema(BaseModel):
    version: str
    graph: str
    description: str = ""
    node_types: dict[str, NodeType]
    edge_types: dict[str, EdgeType]
    edge_global_props: dict[str, str]
    meta: SchemaMeta

    @model_validator(mode="after")
    def _check_edge_endpoints(self) -> "KGSchema":
        valid = set(self.node_types) | {ANY_NODE}
        for name, edge in self.edge_types.items():
            if edge.head_type not in valid:
                raise ValueError(
                    f"Edge {name}: head_type '{edge.head_type}' is not a declared node type"
                )
            if edge.tail_type not in valid:
                raise ValueError(
                    f"Edge {name}: tail_type '{edge.tail_type}' is not a declared node type"
                )
        return self

    @model_validator(mode="after")
    def _check_meta_counts(self) -> "KGSchema":
        if self.meta.total_node_types != len(self.node_types):
            raise ValueError(
                f"meta.total_node_types={self.meta.total_node_types} "
                f"!= len(node_types)={len(self.node_types)}"
            )
        if self.meta.total_edge_types != len(self.edge_types):
            raise ValueError(
                f"meta.total_edge_types={self.meta.total_edge_types} "
                f"!= len(edge_types)={len(self.edge_types)}"
            )
        if self.meta.primekg_inherited + self.meta.clinical_extension != self.meta.total_edge_types:
            raise ValueError(
                "meta: primekg_inherited + clinical_extension must equal total_edge_types"
            )
        return self

    @model_validator(mode="after")
    def _check_global_edge_props(self) -> "KGSchema":
        missing = REQUIRED_GLOBAL_EDGE_PROPS - set(self.edge_global_props)
        if missing:
            raise ValueError(f"edge_global_props missing required keys: {sorted(missing)}")
        return self


DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "config" / "kg_schema.yaml"


def load_schema(path: str | Path = DEFAULT_SCHEMA_PATH) -> KGSchema:
    """Load and validate the PrimeKG-X schema from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return KGSchema.model_validate(data)
