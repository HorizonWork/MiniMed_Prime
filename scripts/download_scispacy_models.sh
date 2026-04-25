#!/usr/bin/env bash
# Install scispacy models via pip. scispacy models aren't on PyPI; they ship
# as tarballs on the AI2 S3 bucket. We pin a known-working release.
#
# Override SCISPACY_MODEL_VERSION / SCISPACY_MODEL_NAME to install a
# different model (e.g. `en_core_sci_sm` for lighter smoke tests).
set -euo pipefail

SCISPACY_VERSION="${SCISPACY_MODEL_VERSION:-0.5.4}"
MODEL="${SCISPACY_MODEL_NAME:-en_core_sci_lg}"
URL="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v${SCISPACY_VERSION}/${MODEL}-${SCISPACY_VERSION}.tar.gz"

echo "scispacy: installing ${MODEL} (v${SCISPACY_VERSION})"
echo "  url: $URL"

if command -v uv >/dev/null 2>&1; then
    uv pip install "$URL"
else
    pip install "$URL"
fi

echo "scispacy: done. The UMLS linker KB will be downloaded on first use "
echo "  (~1 GB) and cached in scispacy's data dir."
