"""MRSTY column order per the UMLS RRF spec.

- CUI:  concept unique identifier
- TUI:  semantic type unique identifier (e.g., T121)
- STN:  semantic type tree number
- STY:  semantic type name (e.g., "Pharmacologic Substance")
- ATUI: attribute unique identifier
- CVF:  content view flag
"""

from __future__ import annotations

MRSTY_COLUMNS: tuple[str, ...] = (
    "CUI",
    "TUI",
    "STN",
    "STY",
    "ATUI",
    "CVF",
)
