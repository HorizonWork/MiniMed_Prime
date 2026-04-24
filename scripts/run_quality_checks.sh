#!/usr/bin/env bash
set -euo pipefail
biomedkg evaluate all --graph-version "${GRAPH_VERSION:-kg_local}"
