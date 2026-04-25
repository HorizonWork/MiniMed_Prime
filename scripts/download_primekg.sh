#!/usr/bin/env bash
# Download the PrimeKG edge CSV from Harvard Dataverse.
# Idempotent: skips download if the destination already exists.
# Set FORCE=1 to redownload, RELEASE=<tag> to override "local".
set -euo pipefail

RELEASE="${RELEASE:-local}"
DATA_DIR="${PRIMEKG_DATA_DIR:-data/primekg}/$RELEASE"
URL="${PRIMEKG_DOWNLOAD_URL:-https://dataverse.harvard.edu/api/access/datafile/6180626}"
DEST="$DATA_DIR/kg.csv"

mkdir -p "$DATA_DIR"

if [[ -f "$DEST" && "${FORCE:-0}" != "1" ]]; then
    echo "primekg: $DEST already exists ($(du -h "$DEST" | cut -f1)). FORCE=1 to redownload."
    exit 0
fi

echo "primekg: downloading"
echo "  from: $URL"
echo "  to:   $DEST"
curl -L --fail --progress-bar --retry 3 --retry-delay 5 -o "$DEST.tmp" "$URL"
mv "$DEST.tmp" "$DEST"
echo "primekg: saved $(du -h "$DEST" | cut -f1)"
