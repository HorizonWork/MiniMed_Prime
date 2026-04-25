-- Phase 4 Knowledge Graph schema extensions.
-- Idempotent; safe to apply on top of 001-005.

-- Fast term-normalized-text lookup for the entity linker fallback tier.
CREATE INDEX IF NOT EXISTS idx_term_normalized_text ON term(normalized_text);
CREATE INDEX IF NOT EXISTS idx_term_concept_id ON term(concept_id);

-- UMLS (SAB, CODE) → CUI crosswalk. Populated by the MRCONSO ingest;
-- used by kg_build.umls_crosswalk to backfill Entity.canonical_cui.
CREATE TABLE IF NOT EXISTS umls_crosswalk (
    sab TEXT NOT NULL,
    code TEXT NOT NULL,
    cui TEXT NOT NULL,
    PRIMARY KEY (sab, code, cui)
);
CREATE INDEX IF NOT EXISTS idx_umls_crosswalk_sab_code ON umls_crosswalk(sab, code);
CREATE INDEX IF NOT EXISTS idx_umls_crosswalk_cui ON umls_crosswalk(cui);

-- Semantic-type attachment table (UMLS MRSTY). Separate from the `semantic_type`
-- vocabulary table so we can store the full (cui, tui, sty) triples efficiently.
CREATE TABLE IF NOT EXISTS concept_semantic_type (
    cui TEXT NOT NULL,
    tui TEXT NOT NULL,
    sty TEXT NOT NULL,
    PRIMARY KEY (cui, tui)
);
CREATE INDEX IF NOT EXISTS idx_concept_sty ON concept_semantic_type(sty);
