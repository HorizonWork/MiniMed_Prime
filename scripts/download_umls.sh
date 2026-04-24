#!/usr/bin/env bash
set -euo pipefail
biomedkg ingest umls --release "${RELEASE:-local}"
