#!/usr/bin/env bash
set -euo pipefail
minimed ingest pubmed --mode baseline --release "${RELEASE:-local}"
