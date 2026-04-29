#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11.15}"

if command -v uv >/dev/null 2>&1; then
  uv python install "$PYTHON_VERSION"
  uv venv --python "$PYTHON_VERSION" .venv
  uv pip install -r requirements.txt -r requirements-dev.txt
else
  python3.11 -m venv .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt -r requirements-dev.txt
fi
