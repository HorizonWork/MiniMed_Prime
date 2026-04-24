"""Pseudocode Airflow DAG: train_trm_dag.py."""
from __future__ import annotations


DAG_ID = "train_trm_dag"


def build_dag():
    # 1. generate_reasoning_paths
    # 2. serialize_examples
    # 3. train_trm
    # 4. evaluate_trm
    return {"dag_id": DAG_ID, "tasks": ['generate_reasoning_paths', 'serialize_examples', 'train_trm', 'evaluate_trm']}


dag = build_dag()
