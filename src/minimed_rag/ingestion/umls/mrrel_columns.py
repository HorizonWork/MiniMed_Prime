"""MRREL column order per the UMLS Rich Release Format (RRF) spec.

- CUI1/AUI1: subject concept / atom
- STYPE1:    source type (CUI, AUI, …)
- REL:       relation class (PAR, CHD, RN, RB, RO, SY, AQ, QB, SIB)
- RELA:      specific relation attribute (may_treat, isa, causative_agent_of, …)
- CUI2/AUI2: object concept / atom
- STYPE2:    source type for object
- RUI:       relation unique identifier (stable assertion-record ID)
- SRUI:      source RUI
- SAB:       source vocabulary
- SL:        source of label
- RG:        relation group
- DIR:       directionality (Y/N/'')
- SUPPRESS:  keep N only
- CVF:       content view flag
"""

from __future__ import annotations

MRREL_COLUMNS: tuple[str, ...] = (
    "CUI1",
    "AUI1",
    "STYPE1",
    "REL",
    "CUI2",
    "AUI2",
    "STYPE2",
    "RELA",
    "RUI",
    "SRUI",
    "SAB",
    "SL",
    "RG",
    "DIR",
    "SUPPRESS",
    "CVF",
)
