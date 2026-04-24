#!/usr/bin/env bash
set -euo pipefail
minimed evaluate all --graph-version "${GRAPH_VERSION:-kg_local}"
