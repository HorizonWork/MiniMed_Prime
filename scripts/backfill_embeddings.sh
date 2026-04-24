#!/usr/bin/env bash
set -euo pipefail
biomedkg build-indexes vector --graph-version "${GRAPH_VERSION:-kg_local}"
