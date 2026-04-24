"""Pseudocode canonical assertion builder."""

from __future__ import annotations

from minimed_rag.common.exceptions import InvalidAssertionError
from minimed_rag.common.ids import make_assertion_id
from minimed_rag.common.time import current_graph_version
from minimed_rag.domain.assertions import Assertion


class AssertionBuilder:
    def __init__(self, entity_repo, predicate_mapper, predicate_registry):
        self.entity_repo = entity_repo
        self.predicate_mapper = predicate_mapper
        self.predicate_registry = predicate_registry

    def build_from_source_assertion(self, source_assertion) -> Assertion:
        subject = self.entity_repo.get_canonical_by_source_id(source_assertion.subject_source_id)
        obj = self.entity_repo.get_canonical_by_source_id(source_assertion.object_source_id)
        predicate = self.predicate_mapper.map_source_predicate(
            source_assertion.source_system,
            source_assertion.source_predicate,
            subject.entity_type,
            obj.entity_type,
        )
        if not self.predicate_registry.is_valid_domain_range(
            predicate, subject.entity_type, obj.entity_type
        ):
            raise InvalidAssertionError(predicate, subject.entity_type, obj.entity_type)
        assertion_id = make_assertion_id(
            subject.entity_id,
            predicate,
            obj.entity_id,
            source_assertion.source_system,
            source_assertion.source_record_id,
        )
        return Assertion(
            assertion_id=assertion_id,
            subject_id=subject.entity_id,
            predicate=predicate,
            object_id=obj.entity_id,
            source_system=source_assertion.source_system,
            source_record_id=source_assertion.source_record_id,
            source_predicate=source_assertion.source_predicate,
            source_release=source_assertion.source_release,
            graph_version=current_graph_version(),
            is_current=True,
        )
