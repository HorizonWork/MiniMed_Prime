"""Pseudocode Airflow DAG: build_indexes_dag.py."""
from __future__ import annotations


DAG_ID = "build_indexes_dag"


def build_dag():
    # 1. embed_chunks
    # 2. load_milvus
    # 3. load_opensearch
    # 4. apply_neo4j_indexes
    return {"dag_id": DAG_ID, "tasks": ['embed_chunks', 'load_milvus', 'load_opensearch', 'apply_neo4j_indexes']}


dag = build_dag()
