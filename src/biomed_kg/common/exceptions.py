"""Pseudocode exceptions raised by pipeline boundaries."""
from __future__ import annotations


class BiomedKGError(Exception):
    pass


class InvalidAssertionError(BiomedKGError):
    def __init__(self, predicate: str, subject_type: str, object_type: str):
        super().__init__(f"Invalid assertion: {subject_type} -[{predicate}]-> {object_type}")


class SourceIngestionError(BiomedKGError):
    pass


class EntityLinkingError(BiomedKGError):
    pass


class RetrievalPlanningError(BiomedKGError):
    pass
