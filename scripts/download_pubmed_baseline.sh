#!/usr/bin/env bash
set -euo pipefail
biomedkg ingest pubmed --mode baseline --release "${RELEASE:-local}"
