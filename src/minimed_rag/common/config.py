"""Runtime configuration.

Two complementary surfaces:

- ``AppConfig`` + ``load_config``: YAML-driven pipeline config
  (``configs/env/{env}.yaml``). Drives pipeline parameters, feature flags,
  source/retrieval/training overrides.
- ``Settings``: pydantic-settings model that reads ``.env`` and process
  environment. Drives storage client wiring (Neo4j/Postgres/etc.) and Phase 4
  KG data paths. Use ``get_settings()`` (cached) for shared access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(slots=True)
class AppConfig:
    env: str
    graph_version: str
    values: dict[str, Any] = field(default_factory=dict)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        current: Any = self.values
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(env: str = "local", config_dir: str | Path = "configs") -> AppConfig:
    raw = load_yaml(Path(config_dir) / "env" / f"{env}.yaml")
    return AppConfig(
        env=str(raw.get("env", env)),
        graph_version=str(raw.get("graph_version", "kg_local")),
        values=raw,
    )


class Settings(BaseSettings):
    """Environment-driven runtime settings (``.env`` and process env).

    Storage factories (``neo4j.connect``, ``postgres.connect``) and CLI
    dependency wiring read from here. Keep defaults aligned with
    ``docker-compose.yml`` so a fresh checkout "just works" after ``make up``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: str = Field(default="local")
    graph_version: str = Field(default="kg_local")

    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="minimed")
    neo4j_database: str = Field(default="neo4j")

    postgres_dsn: str = Field(default="postgresql://minimed:minimed@localhost:5432/minimed")

    minio_endpoint: str = Field(default="http://localhost:9000")
    minio_access_key: str = Field(default="minio")
    minio_secret_key: str = Field(default="minio123")

    milvus_uri: str = Field(default="http://localhost:19530")
    opensearch_url: str = Field(default="http://localhost:9200")

    primekg_data_dir: str = Field(default="data/primekg")
    primekg_download_url: str = Field(
        default="https://dataverse.harvard.edu/api/access/datafile/6180626"
    )
    umls_data_dir: str = Field(default="data/umls")
    scispacy_model: str = Field(default="en_core_sci_lg")
    scispacy_umls_linker: bool = Field(default=True)

    def primekg_csv_path(self, release: str) -> Path:
        return Path(self.primekg_data_dir) / release / "kg.csv"

    def umls_release_dir(self, release: str) -> Path:
        return Path(self.umls_data_dir) / release


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
