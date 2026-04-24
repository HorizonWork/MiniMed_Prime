#!/usr/bin/env bash
set -euo pipefail
minimed ingest umls --release "${RELEASE:-local}"
