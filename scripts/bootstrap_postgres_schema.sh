#!/usr/bin/env bash
# Apply the canonical Postgres schema idempotently.
# Reads POSTGRES_DSN from env (defaults to the compose-default DSN).
set -euo pipefail

DSN="${POSTGRES_DSN:-postgresql://minimed:minimed@localhost:5432/minimed}"
SQL_DIR="$(cd "$(dirname "$0")/../data_contracts/sql" && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

echo "Applying Postgres schema from $SQL_DIR"
echo "  DSN: $DSN"

shopt -s nullglob
if command -v psql >/dev/null 2>&1; then
    for path in "$SQL_DIR"/*.sql; do
        file=$(basename "$path")
        echo "  -> $file"
        psql "$DSN" -v ON_ERROR_STOP=1 -q -f "$path"
    done
else
    echo "  psql not found; applying schema via psycopg"
    uv run python - "$DSN" "$SQL_DIR" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import psycopg


dsn = sys.argv[1]
sql_dir = Path(sys.argv[2])

with psycopg.connect(dsn, autocommit=False) as conn:
    for path in sorted(sql_dir.glob("*.sql")):
        print(f"  -> {path.name}", flush=True)
        with path.open("r", encoding="utf-8") as fh:
            conn.execute(fh.read())
        conn.commit()
PY
fi

echo "Done."
