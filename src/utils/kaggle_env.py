from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - torch is part of the project stack
    torch = None


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())


class KaggleEnv:
    """Auto-detects Kaggle vs local execution and resolves project paths."""

    INPUT_ROOT = Path("/kaggle/input")
    WORKING_ROOT = Path("/kaggle/working")
    LOCAL_ASSET_ROOT = Path(
        os.environ.get(
            "MINIMED_KAGGLE_ASSETS", "/home/n91ym1nhky/Projects/ext-minimed/data"
        )
    )
    DATASET_MOUNT_ALIASES = {
        "data/medreason": "medreason",
        "data/datasets/medreason": "medreason",
        "data/kg/primekg": "primekg",
        "data/checkpoints/sapbert": "sapbert",
        "data/checkpoints/medcpt-query": "medcpt-query",
        "data/checkpoints/medcpt-cross": "medcpt-cross",
        "data/checkpoints/medcpt-article": "medcpt-article",
        "data/checkpoints/medcpt/query_encoder": "medcpt-query",
        "data/checkpoints/medcpt/cross_encoder": "medcpt-cross",
        "data/checkpoints/medcpt/article_encoder": "medcpt-article",
        "data/checkpoints/medreason-8b": "medreason-8b",
        "data/checkpoints/llm": "medreason-8b",
        "external/TinyRecursiveModels": "trm-real",
    }
    DATASET_SLUG_HINTS = {
        "primekg": ("primekg-medical-knowledge-graph",),
        "medreason": ("medreason-medical-reasoning-dataset",),
        "sapbert": ("sapbert-from-pubmedbert-fulltext",),
        "medcpt-query": ("medcpt-query-encoder",),
        "medcpt-cross": ("medcpt-cross-encoder",),
        "medcpt-article": ("medcpt-article-encoder",),
        "trm-real": ("tinyrecursivemodels-real-repo-and-arc", "tinyrecursivemodels",),
    }
    DATASET_SIGNATURES = {
        "primekg": (("edges.csv", "nodes.csv"), ("kg.csv",)),
        "medreason": (("ours_quality_33000.jsonl",),),
        "sapbert": (("config.json", "vocab.txt"), ("config.json", "tokenizer_config.json")),
        "medcpt-query": (("config.json", "tokenizer_config.json"),),
        "medcpt-cross": (("config.json", "tokenizer_config.json"),),
        "medcpt-article": (("config.json", "tokenizer_config.json"),),
        "medreason-8b": (("config.json",),),
        "trm-real": (("models/recursive_reasoning/trm.py",),),
    }

    @staticmethod
    def is_kaggle() -> bool:
        return os.path.exists("/kaggle/input")

    @classmethod
    def path(cls, local_relative: str | Path) -> Path:
        path = Path(local_relative)
        if path.is_absolute():
            return path

        normalized = str(path).replace("\\", "/").lstrip("./")
        if not cls.is_kaggle():
            local_asset = cls._resolve_known_local_asset(normalized)
            if local_asset is not None and (not path.exists() or cls._is_empty_dir(path)):
                return local_asset
            return path

        if normalized.startswith("external/TinyRecursiveModels"):
            suffix = normalized[len("external/TinyRecursiveModels") :].lstrip("/")
            working_trm = cls.WORKING_ROOT / "external" / "TinyRecursiveModels"
            resolved_working_trm = cls._resolve_trm_repo_mount(working_trm, suffix)
            if resolved_working_trm is not None:
                return resolved_working_trm
        elif normalized.startswith("external/"):
            working_external = cls.WORKING_ROOT / Path(normalized)
            if working_external.exists():
                return working_external
        mounted_alias = cls._resolve_known_input_mount(normalized)
        if mounted_alias is not None:
            return mounted_alias
        if normalized.startswith("external/"):
            return cls.WORKING_ROOT / Path(normalized)
        if normalized.startswith("data/logs"):
            return cls.WORKING_ROOT / "logs" / Path(normalized).relative_to("data/logs")
        if normalized.startswith("data/checkpoints/active"):
            return cls.WORKING_ROOT / "checkpoints" / "active" / Path(normalized).relative_to("data/checkpoints/active")
        if normalized in {"data/pubmed_cache.jsonl", "data/smoke_test_output.json"}:
            return cls.WORKING_ROOT / Path(normalized).name
        if normalized.startswith("data/") or normalized.startswith("notebooks/"):
            resolved_input = cls._search_input(path)
            if resolved_input is not None:
                return resolved_input
            if normalized.startswith("data/checkpoints"):
                return cls.WORKING_ROOT / "checkpoints" / Path(normalized).relative_to("data/checkpoints")
            if normalized.startswith("data/eval"):
                return cls.WORKING_ROOT / "eval" / Path(normalized).relative_to("data/eval")
            return cls.WORKING_ROOT / Path(normalized).relative_to("data")
        return path

    @classmethod
    def ensure_writeable(cls, path: Path) -> Path:
        resolved = path if path.is_absolute() else cls.path(path)
        if cls.is_kaggle():
            try:
                resolved.relative_to(cls.WORKING_ROOT)
            except ValueError:
                tail = path if not path.is_absolute() else Path(path.name)
                if str(tail).replace("\\", "/").startswith("data/"):
                    resolved = cls.WORKING_ROOT / Path(str(tail).replace("\\", "/")).relative_to("data")
                else:
                    resolved = cls.WORKING_ROOT / tail.name
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    @classmethod
    def _resolve_known_input_mount(cls, normalized: str) -> Path | None:
        for prefix, dataset_slug in sorted(cls.DATASET_MOUNT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
            if normalized != prefix and not normalized.startswith(prefix + "/"):
                continue
            suffix = normalized[len(prefix):].lstrip("/")
            candidate = cls._find_input_dataset_root(dataset_slug)
            if candidate is None:
                continue
            if dataset_slug == "trm-real":
                resolved_trm = cls._resolve_trm_repo_mount(candidate, suffix)
                if resolved_trm is not None:
                    return resolved_trm
            if suffix:
                candidate = candidate / suffix
            if candidate.exists():
                return candidate
        return None

    @classmethod
    def _resolve_known_local_asset(cls, normalized: str) -> Path | None:
        for prefix, dataset_slug in sorted(cls.DATASET_MOUNT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
            if normalized != prefix and not normalized.startswith(prefix + "/"):
                continue
            suffix = normalized[len(prefix):].lstrip("/")
            candidates = cls._local_asset_candidates(dataset_slug)
            for candidate in candidates:
                resolved = candidate / suffix if suffix else candidate
                if dataset_slug == "trm-real":
                    trm_resolved = cls._resolve_trm_repo_mount(candidate, suffix)
                    if trm_resolved is not None:
                        return trm_resolved
                if resolved.exists():
                    return resolved
        return None

    @classmethod
    def _local_asset_candidates(cls, dataset_slug: str) -> tuple[Path, ...]:
        if dataset_slug == "primekg":
            return (
                cls.LOCAL_ASSET_ROOT / "primekg-upload",
                cls.LOCAL_ASSET_ROOT / "primekg",
                cls.LOCAL_ASSET_ROOT / "kg" / "primekg",
            )
        if dataset_slug == "medreason":
            return (
                cls.LOCAL_ASSET_ROOT / "medreason-upload",
                cls.LOCAL_ASSET_ROOT / "medreason",
                cls.LOCAL_ASSET_ROOT / "datasets" / "medreason",
            )
        if dataset_slug == "trm-real":
            return (
                cls.LOCAL_ASSET_ROOT / "checkpoints" / "trm",
                cls.LOCAL_ASSET_ROOT / "trm-real",
                cls.LOCAL_ASSET_ROOT / "TinyRecursiveModels",
            )
        return (
            cls.LOCAL_ASSET_ROOT / "checkpoints" / dataset_slug,
            cls.LOCAL_ASSET_ROOT / dataset_slug,
        )

    @staticmethod
    def _is_empty_dir(path: Path) -> bool:
        if not path.is_dir():
            return False
        try:
            next(path.iterdir())
            return False
        except StopIteration:
            return True
        except OSError:
            return False

    @classmethod
    def _resolve_trm_repo_mount(cls, root: Path, suffix: str = "", allow_direct: bool = False) -> Path | None:
        if not root.exists():
            return None
        repo_root = cls._find_trm_repo_root(root)
        if repo_root is not None:
            resolved = repo_root / suffix if suffix else repo_root
            if resolved.exists():
                return resolved
        direct = root / suffix if suffix else root
        return direct if allow_direct and direct.exists() else None

    @classmethod
    def _find_trm_repo_root(cls, root: Path) -> Path | None:
        candidate_roots = (
            root,
            root / "TinyRecursiveModels",
            root / "TinyRecursiveModels_bundle" / "TinyRecursiveModels",
            root / "external" / "TinyRecursiveModels",
        )
        for candidate in candidate_roots:
            if (candidate / "models" / "recursive_reasoning" / "trm.py").exists():
                return candidate

        try:
            trm_file = next(root.glob("**/models/recursive_reasoning/trm.py"))
        except (StopIteration, OSError):
            return None
        return trm_file.parents[2]

    @classmethod
    def _search_input(cls, path: Path) -> Path | None:
        if not cls.is_kaggle() or not cls.INPUT_ROOT.exists():
            return None

        candidates: list[Path] = []
        normalized = str(path).replace("\\", "/")
        if normalized.startswith("data/"):
            relative = Path(normalized).relative_to("data")
            candidates.extend(
                [
                    cls.INPUT_ROOT / relative,
                    cls.INPUT_ROOT / relative.name,
                ]
            )
        candidates.append(cls.INPUT_ROOT / path.name)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        dataset_dirs = list(cls._iter_input_dataset_dirs())
        search_suffixes = []
        if normalized.startswith("data/"):
            relative = Path(normalized).relative_to("data")
            search_suffixes.extend(
                [
                    relative,
                    Path(relative.name),
                    Path(*relative.parts[-2:]) if len(relative.parts) >= 2 else Path(relative.name),
                ]
            )
        else:
            search_suffixes.append(Path(path.name))

        for dataset_dir in dataset_dirs:
            for suffix in search_suffixes:
                if cls._slugify(dataset_dir.name) == cls._slugify(suffix.name):
                    return dataset_dir
                candidate = dataset_dir / suffix
                if candidate.exists():
                    return candidate
        return None

    @classmethod
    def _iter_input_dataset_dirs(cls) -> tuple[Path, ...]:
        if not cls.INPUT_ROOT.exists():
            return ()
        discovered: list[Path] = []
        seen: set[Path] = set()

        def register(path: Path) -> None:
            if not path.is_dir():
                return
            resolved = path.resolve()
            if resolved in seen:
                return
            seen.add(resolved)
            discovered.append(path)

        for entry in cls.INPUT_ROOT.iterdir():
            if not entry.is_dir():
                continue
            if entry.name != "datasets":
                register(entry)

        nested_root = cls.INPUT_ROOT / "datasets"
        if nested_root.exists():
            for owner_dir in nested_root.iterdir():
                if not owner_dir.is_dir():
                    continue
                for dataset_dir in owner_dir.iterdir():
                    if not dataset_dir.is_dir():
                        continue
                    versions_dir = dataset_dir / "versions"
                    if versions_dir.is_dir():
                        for version_dir in versions_dir.iterdir():
                            register(version_dir)
                    register(dataset_dir)

        return tuple(discovered)

    @classmethod
    def _find_input_dataset_root(cls, dataset_slug: str) -> Path | None:
        direct = cls.INPUT_ROOT / dataset_slug
        if direct.exists():
            return direct

        versioned = cls._find_versioned_input_dataset_root(dataset_slug)
        if versioned is not None:
            return versioned

        for dataset_dir in cls._iter_input_dataset_dirs():
            if cls._dataset_dir_matches_slug(dataset_dir, dataset_slug):
                return dataset_dir

        if dataset_slug == "trm-real":
            for dataset_dir in cls._iter_input_dataset_dirs():
                repo_root = cls._find_trm_repo_root(dataset_dir)
                if repo_root is not None:
                    return repo_root

        for dataset_dir in cls._iter_input_dataset_dirs():
            if cls._dataset_dir_matches_signature(dataset_dir, dataset_slug):
                return dataset_dir
        return None

    @classmethod
    def _find_versioned_input_dataset_root(cls, dataset_slug: str) -> Path | None:
        nested_root = cls.INPUT_ROOT / "datasets"
        if not nested_root.exists():
            return None

        candidates: list[Path] = []
        for owner_dir in nested_root.iterdir():
            if not owner_dir.is_dir():
                continue
            for dataset_dir in owner_dir.iterdir():
                if not dataset_dir.is_dir() or not cls._dataset_dir_matches_slug(dataset_dir, dataset_slug):
                    continue
                versions_dir = dataset_dir / "versions"
                if not versions_dir.is_dir():
                    continue
                version_dirs = [path for path in versions_dir.iterdir() if path.is_dir()]
                version_dirs.sort(key=lambda path: cls._version_sort_key(path.name), reverse=True)
                candidates.extend(version_dirs)

        for candidate in candidates:
            if cls._dataset_dir_matches_signature(candidate, dataset_slug):
                return candidate
            try:
                next(candidate.iterdir())
                return candidate
            except (StopIteration, OSError):
                continue
        return None

    @classmethod
    def _dataset_dir_matches_slug(cls, dataset_dir: Path, dataset_slug: str) -> bool:
        expected_slug = cls._slugify(dataset_slug)
        candidates = {cls._slugify(dataset_dir.name)}
        metadata = cls._read_dataset_metadata(dataset_dir)
        if metadata:
            dataset_id = metadata.get("id")
            if isinstance(dataset_id, str) and dataset_id.strip():
                candidates.add(cls._slugify(dataset_id.split("/")[-1]))
            title = metadata.get("title")
            if isinstance(title, str) and title.strip():
                candidates.add(cls._slugify(title))
        hints = tuple(cls.DATASET_SLUG_HINTS.get(dataset_slug, ()))
        normalized_expected = (expected_slug, *(cls._slugify(hint) for hint in hints))
        return any(cls._slug_candidate_matches(candidate, normalized_expected) for candidate in candidates)

    @classmethod
    def _dataset_dir_matches_signature(cls, dataset_dir: Path, dataset_slug: str) -> bool:
        signatures = cls.DATASET_SIGNATURES.get(dataset_slug, ())
        if not signatures:
            return False
        for signature in signatures:
            if all((dataset_dir / relative_path).exists() for relative_path in signature):
                return True
        return False

    @staticmethod
    def _slug_candidate_matches(candidate: str, expected_values: tuple[str, ...]) -> bool:
        for expected in expected_values:
            if not expected:
                continue
            if candidate == expected:
                return True
            if candidate.startswith(expected + "-"):
                return True
            if expected.startswith(candidate + "-"):
                return True
            if expected in candidate:
                return True
        return False

    @staticmethod
    def _version_sort_key(value: str) -> tuple[int, str]:
        try:
            return int(value), value
        except ValueError:
            return -1, value

    @staticmethod
    def _read_dataset_metadata(dataset_dir: Path) -> dict[str, Any] | None:
        metadata_path = dataset_dir / "dataset-metadata.json"
        if not metadata_path.exists():
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
        return slug.strip("-")


class T4Hardening:
    @staticmethod
    def setup_memory() -> None:
        if torch is None:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = False

    @staticmethod
    def oom_recovery(model: Any, batch_size: int) -> int:
        del model
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        recovered_batch_size = max(1, batch_size // 2)
        LOGGER.warning("CUDA OOM recovery triggered. Reducing batch size from %s to %s.", batch_size, recovered_batch_size)
        return recovered_batch_size

    @staticmethod
    def verify_fp16(requested_dtype: str | None = None) -> None:
        normalized_dtype = str(requested_dtype or "").lower()
        if normalized_dtype in {"bf16", "bfloat16"}:
            raise ValueError("BF16 is not supported on T4 GPUs. Configure the runtime to use FP16.")
        if torch is None:
            return
        if torch.get_default_dtype() == torch.bfloat16:
            raise ValueError("The global torch default dtype is bfloat16, which is not supported for this T4-targeted pipeline.")
        if not torch.cuda.is_available():
            LOGGER.warning("CUDA is not available. FP16 verification is running in CPU-only compatibility mode.")
            return
        capability = torch.cuda.get_device_capability(0)
        if capability < (7, 0):
            LOGGER.warning("The active GPU compute capability %s is older than the expected T4 baseline.", capability)
        if capability != (7, 5):
            LOGGER.warning("The active GPU capability is %s; the runtime is hardened for Kaggle T4 (7.5).", capability)


__all__ = ["KaggleEnv", "T4Hardening"]
