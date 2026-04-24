"""Pseudocode Airflow DAG: ingest_pubmed_dag.py."""
from __future__ import annotations


DAG_ID = "ingest_pubmed_dag"


def build_dag():
    # 1. download_pubmed
    # 2. parse_xml
    # 3. chunk_documents
    # 4. enqueue_nlp_jobs
    return {"dag_id": DAG_ID, "tasks": ['download_pubmed', 'parse_xml', 'chunk_documents', 'enqueue_nlp_jobs']}


dag = build_dag()
