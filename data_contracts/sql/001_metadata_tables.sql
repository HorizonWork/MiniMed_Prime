CREATE TABLE IF NOT EXISTS source_release (
    source_release_id TEXT PRIMARY KEY,
    source_system TEXT NOT NULL,
    release_name TEXT NOT NULL,
    released_at TIMESTAMPTZ,
    checksum TEXT
);

CREATE TABLE IF NOT EXISTS ingestion_run (
    ingestion_run_id TEXT PRIMARY KEY,
    source_system TEXT NOT NULL,
    source_release_id TEXT REFERENCES source_release(source_release_id),
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS pipeline_run (
    pipeline_run_id TEXT PRIMARY KEY,
    pipeline_name TEXT NOT NULL,
    graph_version TEXT,
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS model_run (
    model_run_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    graph_version TEXT NOT NULL,
    metrics JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS graph_snapshot (
    graph_version TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    entity_count BIGINT DEFAULT 0,
    assertion_count BIGINT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS index_snapshot (
    index_snapshot_id TEXT PRIMARY KEY,
    graph_version TEXT NOT NULL,
    index_type TEXT NOT NULL,
    index_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
