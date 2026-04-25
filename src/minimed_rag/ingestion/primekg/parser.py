"""PrimeKG CSV parser.

Two entry points:

- ``iter_primekg_edges(path, limit=None)`` — streaming generator over the on-disk CSV.
  Required for the ~8M-edge production file; never materialises the full list.
- ``parse_primekg_edges(raw_file)`` — legacy interface that consumes a
  ``RawFile``'s in-memory bytes. Kept for small inputs and unit tests.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from minimed_rag.ingestion.base import RawFile

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RawPrimeKGEdge:
    source_node_id: str
    source_node_name: str
    source_node_type: str
    relation: str
    target_node_id: str
    target_node_name: str
    target_node_type: str
    source: str | None = None


_REQUIRED_COLUMNS = (
    "x_id",
    "x_name",
    "x_type",
    "relation",
    "y_id",
    "y_name",
    "y_type",
)


def _row_to_edge(row: dict[str, str]) -> RawPrimeKGEdge:
    return RawPrimeKGEdge(
        source_node_id=row["x_id"],
        source_node_name=row["x_name"],
        source_node_type=row["x_type"],
        relation=row["relation"],
        target_node_id=row["y_id"],
        target_node_name=row["y_name"],
        target_node_type=row["y_type"],
        source=(row.get("source") or None),
    )


def iter_primekg_edges(
    path: str | Path,
    limit: int | None = None,
    on_error: str = "skip",
) -> Iterator[RawPrimeKGEdge]:
    """Stream PrimeKG edges from a CSV on disk.

    ``on_error="skip"`` logs malformed rows and continues (suitable for the
    full production file where a handful of bad rows shouldn't abort an
    hours-long ingest). ``on_error="raise"`` fails fast (for tests).
    """
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in _REQUIRED_COLUMNS if c not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(
                f"PrimeKG CSV missing required columns: {missing}; got {reader.fieldnames}"
            )
        count = 0
        for row_idx, row in enumerate(reader):
            if limit is not None and count >= limit:
                return
            try:
                yield _row_to_edge(row)
                count += 1
            except KeyError as exc:
                if on_error == "raise":
                    raise
                logger.warning("primekg: skipping row %d: missing %s", row_idx, exc)


def parse_primekg_edges(raw_file: RawFile) -> list[RawPrimeKGEdge]:
    """Legacy interface: parse bytes of the CSV into a list. Small inputs only."""
    reader = csv.DictReader(StringIO(raw_file.bytes.decode("utf-8")))
    return [_row_to_edge(row) for row in reader]
