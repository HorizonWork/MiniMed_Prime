CREATE TABLE IF NOT EXISTS dataset_snapshot (dataset_snapshot_id TEXT PRIMARY KEY, dataset_name TEXT NOT NULL, graph_version TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL);
CREATE TABLE IF NOT EXISTS train_example (train_example_id TEXT PRIMARY KEY, dataset_snapshot_id TEXT REFERENCES dataset_snapshot(dataset_snapshot_id), task_id TEXT NOT NULL, payload JSONB NOT NULL, split TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS entity_vocab (token_id INTEGER PRIMARY KEY, entity_id TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS relation_vocab (token_id INTEGER PRIMARY KEY, predicate TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS serialization_vocab (token_id INTEGER PRIMARY KEY, namespace TEXT NOT NULL, token TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS training_split (task_id TEXT NOT NULL, split TEXT NOT NULL, dataset_snapshot_id TEXT REFERENCES dataset_snapshot(dataset_snapshot_id));
CREATE TABLE IF NOT EXISTS evaluation_result (evaluation_result_id TEXT PRIMARY KEY, graph_version TEXT NOT NULL, evaluator TEXT NOT NULL, metrics JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL);
