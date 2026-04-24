#!/usr/bin/env bash
set -euo pipefail
minimed build-indexes vector --graph-version "${GRAPH_VERSION:-kg_local}"
