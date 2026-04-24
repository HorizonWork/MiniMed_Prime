CREATE INDEX IF NOT EXISTS idx_entity_type ON entity(entity_type);
CREATE INDEX IF NOT EXISTS idx_entity_cui ON entity(canonical_cui);
CREATE INDEX IF NOT EXISTS idx_assertion_spo ON assertion(subject_id, predicate, object_id);
CREATE INDEX IF NOT EXISTS idx_assertion_predicate ON assertion(predicate);
CREATE INDEX IF NOT EXISTS idx_document_chunk_doc ON document_chunk(doc_id);
CREATE INDEX IF NOT EXISTS idx_reasoning_path_task ON reasoning_path(task_id);
CREATE INDEX IF NOT EXISTS idx_train_example_split ON train_example(split);
