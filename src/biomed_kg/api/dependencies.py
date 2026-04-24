"""Pseudocode FastAPI dependency container."""
from __future__ import annotations


class ServiceContainer:
    def __init__(self):
        self.services = {}

    def get(self, name: str):
        return self.services[name]


container = ServiceContainer()


def get_services() -> ServiceContainer:
    return container
