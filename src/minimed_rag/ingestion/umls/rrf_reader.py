"""UMLS RRF reader.

Two entry points:

- ``iter_rrf_lines(path, columns, ...)`` — streaming generator that yields
  dicts keyed by column name. Required for MRREL (~50M rows).
- ``read_rrf_lines(data)`` — legacy bytes-in interface preserved for tests.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

logger = logging.getLogger(__name__)


def read_rrf_lines(data: bytes) -> list[list[str]]:
    text = data.decode("utf-8", errors="replace")
    return [line.rstrip("|\n").split("|") for line in text.splitlines() if line]


def iter_rrf_lines(
    path: str | Path,
    columns: Sequence[str],
    limit: int | None = None,
    on_error: str = "skip",
) -> Iterator[dict[str, str]]:
    """Stream lines from an RRF file, yielding ``{column: value}`` dicts.

    RRF format: pipe-delimited, trailing pipe before newline, no header.
    We strip the trailing ``|\\n`` so the number of fields matches ``columns``
    exactly — lines with a different field count are logged and skipped.
    """
    path = Path(path)
    column_count = len(columns)
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_idx, raw_line in enumerate(fh):
            if limit is not None and line_idx >= limit:
                return
            line = raw_line.rstrip("\n\r")
            if not line:
                continue
            # RRF always ends each record with a trailing `|` before the
            # newline. We keep it so that split() yields the trailing CVF
            # field (which may legitimately be empty) — but we drop the
            # sentinel empty string that results from the split.
            fields = line.split("|")
            if len(fields) == column_count + 1 and fields[-1] == "":
                fields = fields[:-1]
            if len(fields) != column_count:
                if on_error == "raise":
                    raise ValueError(
                        f"RRF line {line_idx} in {path.name}: "
                        f"got {len(fields)} fields, expected {column_count}"
                    )
                if line_idx < 5:
                    logger.warning(
                        "rrf: skipping malformed line %d in %s (got %d fields, expected %d)",
                        line_idx,
                        path.name,
                        len(fields),
                        column_count,
                    )
                continue
            yield dict(zip(columns, fields, strict=True))
