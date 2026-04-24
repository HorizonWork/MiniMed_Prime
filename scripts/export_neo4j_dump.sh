#!/usr/bin/env bash
set -euo pipefail
neo4j-admin database dump neo4j --to-path="${OUT_DIR:-artifacts/exports}"
