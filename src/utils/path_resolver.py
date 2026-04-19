from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from .kaggle_env import KaggleEnv

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())


class KagglePathResolver:
    """Centralized path resolution for Kaggle vs local runtime."""

    SAPBERT_CANDIDATES = [
        "/kaggle/input/medv3-checkpoints/sapbert",
        "/kaggle/input/datasets/huynhnhuthuyk18hcm/sapbert",
        "/kaggle/input/sapbert",
        "data/checkpoints/sapbert",
    ]

    TRM_REPO_CANDIDATES = [
        "/kaggle/input/medv3-checkpoints/trm",
        "/kaggle/working/external/TinyRecursiveModels",
        "/kaggle/input/datasets/huynhnhuthuyk18hcm/tinyrecursivemodels",
        "external/TinyRecursiveModels",
        "data/checkpoints/trm",
    ]

    TRM_CHECKPOINT_CANDIDATES = [
        "/kaggle/input/medv3-checkpoints/trm/trm_arc_v1_public_step_518071.pt",
        "/kaggle/working/external/TinyRecursiveModels/checkpoints/trm_arc_v1_public_step_518071.pt",
        "/kaggle/input/trm/checkpoints/trm_arc_v1_public_step_518071.pt",
        "/kaggle/input/datasets/huynhnhuthuyk18hcm/tinyrecursivemodels/checkpoints/trm_arc_v1_public_step_518071.pt",
        "external/TinyRecursiveModels/checkpoints/trm_arc_v1_public_step_518071.pt",
    ]

    MEDCPT_ARTICLE_CANDIDATES = [
        "/kaggle/input/medv3-checkpoints/medcpt-article",
        "/kaggle/input/datasets/huynhnhuthuyk18hcm/medcpt-article",
        "data/checkpoints/medcpt-article",
    ]
    TRM_SAMSUNG_SOURCE = Path("models/recursive_reasoning/trm.py")
    TRM_LEGACY_SOURCE = Path("src/trm.py")

    @classmethod
    def resolve_sapbert(cls) -> str | None:
        resolved = cls._resolve_first_existing(cls.SAPBERT_CANDIDATES, label="SapBERT")
        return str(resolved) if resolved is not None else None

    @classmethod
    def resolve_trm_repo(cls) -> str | None:
        for root in cls._iter_existing_paths(cls.TRM_REPO_CANDIDATES):
            samsung_source = root / cls.TRM_SAMSUNG_SOURCE
            legacy_source = root / cls.TRM_LEGACY_SOURCE
            if samsung_source.exists():
                logger.info("Found Samsung TRM structure at: %s", root)
                return str(root)
            if legacy_source.exists():
                logger.info("Found legacy TRM structure at: %s", root)
                return str(root)
            logger.warning("Skipping TRM repo candidate with unknown structure: %s", root)
        logger.warning("No valid TRM repo structure found in any candidate path.")
        return None

    @classmethod
    def resolve_trm_checkpoint(cls) -> str | None:
        resolved = cls._resolve_first_existing(cls.TRM_CHECKPOINT_CANDIDATES, label="TRM checkpoint")
        return str(resolved) if resolved is not None else None

    @classmethod
    def resolve_medcpt(cls) -> str | None:
        resolved = cls._resolve_first_existing(cls.MEDCPT_ARTICLE_CANDIDATES, label="MedCPT article")
        return str(resolved) if resolved is not None else None

    @classmethod
    def get_trm_import_path(cls, repo_root: str | Path) -> tuple[str, str]:
        root = Path(repo_root)
        samsung_source = root / cls.TRM_SAMSUNG_SOURCE
        legacy_source = root / cls.TRM_LEGACY_SOURCE
        if samsung_source.exists():
            return "models.recursive_reasoning.trm", "TinyRecursiveModel"
        if legacy_source.exists():
            return "src.trm", "TinyRecursiveModel"
        raise FileNotFoundError(
            f"No TRM implementation found in {root}. "
            f"Expected {cls.TRM_SAMSUNG_SOURCE} or {cls.TRM_LEGACY_SOURCE}."
        )

    @classmethod
    def _resolve_first_existing(cls, candidates: Iterable[str | Path], *, label: str) -> Path | None:
        for path in cls._iter_existing_paths(candidates):
            logger.info("Resolved %s: %s", label, path)
            return path
        logger.warning("%s not found in any candidate path.", label)
        return None

    @classmethod
    def _iter_existing_paths(cls, candidates: Iterable[str | Path]) -> Iterable[Path]:
        seen: set[Path] = set()
        for candidate in candidates:
            candidate_path = Path(candidate)
            path_variants: list[Path] = []
            if candidate_path.is_absolute():
                path_variants.append(candidate_path)
            else:
                path_variants.append(KaggleEnv.path(candidate_path))
                path_variants.append(candidate_path)
            for path in path_variants:
                resolved = path.resolve(strict=False)
                if resolved in seen:
                    continue
                seen.add(resolved)
                if path.exists():
                    yield path


__all__ = ["KagglePathResolver"]
