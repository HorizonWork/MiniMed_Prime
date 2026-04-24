"""Pseudocode Airflow DAG: ingest_umls_dag.py."""
from __future__ import annotations


DAG_ID = "ingest_umls_dag"


def build_dag():
    # 1. download_umls
    # 2. parse_rrf
    # 3. normalize_terminology
    # 4. write_silver_tables
    return {"dag_id": DAG_ID, "tasks": ['download_umls', 'parse_rrf', 'normalize_terminology', 'write_silver_tables']}


dag = build_dag()
