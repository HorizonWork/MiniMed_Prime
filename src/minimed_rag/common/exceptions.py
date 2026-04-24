"""Pseudocode exceptions raised by pipeline boundaries."""

from __future__ import annotations


class MinimedKGError(Exception):
    pass


class InvalidAssertionError(MinimedKGError):
    def __init__(self, predicate: str, subject_type: str, object_type: str):
        super().__init__(f"Invalid assertion: {subject_type} -[{predicate}]-> {object_type}")


class SourceIngestionError(MinimedKGError):
    pass


class EntityLinkingError(MinimedKGError):
    pass


class RetrievalPlanningError(MinimedKGError):
    pass
