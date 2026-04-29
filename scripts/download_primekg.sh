#!/usr/bin/env bash
# Download PrimeKG v2.1 from Harvard Dataverse (DOI 10.7910/DVN/IXA7BM).
# Resumable; safe to re-run. Files land in kg/data/raw/.
#
# Files (~990 MB total):
#   nodes.tab  (~9 MB)   — 129K nodes with ids, names, types
#   kg.csv     (~982 MB) — full edge list (4M rows) with relation, head, tail, source
#
# Usage:  bash scripts/download_primekg.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/kg/data/raw"
BASE="https://dataverse.harvard.edu/api/access/datafile"

mkdir -p "$DEST"
cd "$DEST"

# nodes.tab — ID 6180617. Pass format=original so Dataverse doesn't auto-convert.
echo "[1/2] nodes.tab"
curl -fL -C - -o nodes.tab "${BASE}/6180617?format=original"

# kg.csv — ID 6180620
echo "[2/2] kg.csv"
curl -fL -C - -o kg.csv "${BASE}/6180620"

echo "Done. Files in: $DEST"
ls -lh "$DEST"
