"""Pseudocode Airflow DAG: build_kg_dag.py."""
from __future__ import annotations


DAG_ID = "build_kg_dag"


def build_dag():
    # 1. resolve_entities
    # 2. map_predicates
    # 3. build_assertions
    # 4. calculate_confidence
    # 5. project_neo4j
    return {"dag_id": DAG_ID, "tasks": ['resolve_entities', 'map_predicates', 'build_assertions', 'calculate_confidence', 'project_neo4j']}


dag = build_dag()
