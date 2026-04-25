"""MRCONSO column order per the UMLS Rich Release Format (RRF) spec.

Field reference:
- CUI:      concept unique identifier
- LAT:      language (ENG, SPA, FRE, …)
- TS:       term status (P=preferred)
- LUI:      term unique identifier
- STT:      string type (PF=preferred form)
- SUI:      string unique identifier
- ISPREF:   Y if the atom is preferred for its CUI
- AUI:      atom unique identifier (stable term_id)
- SAUI:     source atom unique identifier
- SCUI:     source concept unique identifier
- SDUI:     source descriptor unique identifier
- SAB:      source vocabulary abbreviation (MSH, MONDO, DRUGBANK, …)
- TTY:      term type (PT=preferred term, SY=synonym, …)
- CODE:     code in the source vocabulary
- STR:      the actual term/string
- SRL:      source restriction level
- SUPPRESS: O/E/Y/N suppression flag — keep N only
- CVF:      content view flag (unused by us)
"""

from __future__ import annotations

MRCONSO_COLUMNS: tuple[str, ...] = (
    "CUI",
    "LAT",
    "TS",
    "LUI",
    "STT",
    "SUI",
    "ISPREF",
    "AUI",
    "SAUI",
    "SCUI",
    "SDUI",
    "SAB",
    "TTY",
    "CODE",
    "STR",
    "SRL",
    "SUPPRESS",
    "CVF",
)
