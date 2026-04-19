from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable

import requests

try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
except Exception:  # pragma: no cover - packaging ships with pip in normal environments
    Requirement = None  # type: ignore[assignment]
    canonicalize_name = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.kaggle_env import KaggleEnv, T4Hardening
from src.utils.structured_logger import DEFAULT_LOG_DIR, StructuredLogger

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())


def _resolve_asset_dir(logical_path: str | Path) -> Path:
    logical = Path(logical_path)
    resolved = KaggleEnv.path(logical)
    if resolved.exists():
        return resolved
    return KaggleEnv.ensure_writeable(logical)


DATA_DIR = KaggleEnv.ensure_writeable(Path("data"))
CHECKPOINT_DIR = KaggleEnv.ensure_writeable(Path("data/checkpoints"))
KG_DIR = KaggleEnv.ensure_writeable(Path("data/kg"))
PRIMEKG_DIR = _resolve_asset_dir("data/kg/primekg")
SAPBERT_DIR = _resolve_asset_dir("data/checkpoints/sapbert")
MEDCPT_QUERY_DIR = _resolve_asset_dir("data/checkpoints/medcpt-query")
MEDCPT_ARTICLE_DIR = _resolve_asset_dir("data/checkpoints/medcpt-article")
MEDCPT_CROSS_DIR = _resolve_asset_dir("data/checkpoints/medcpt-cross")
SYNTHESIS_MODEL_DIR = _resolve_asset_dir("data/checkpoints/medreason-8b")
EXTERNAL_DIR = KaggleEnv.ensure_writeable(KaggleEnv.path("external") if KaggleEnv.is_kaggle() else PROJECT_ROOT / "external")
TRM_REPO_DIR = EXTERNAL_DIR / "TinyRecursiveModels"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements-integration.txt"

DATAVERSE_BASE_URL = "https://dataverse.harvard.edu"
PRIMEKG_PERSISTENT_ID = "doi:10.7910/DVN/IXA7BM"
SAMSUNG_TRM_REPO_URL = "https://github.com/SamsungSAILMontreal/TinyRecursiveModels"
TRM_KAGGLE_DATASET_SLUG = "trm-real"
TRM_ARCHIVE_NAMES = (
    "TinyRecursiveModels_bundle.tar",
    "TinyRecursiveModels_bundle.zip",
)
SAPBERT_REPO_ID = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
MEDCPT_REPO_IDS = {
    "medcpt-query": "ncbi/MedCPT-Query-Encoder",
    "medcpt-article": "ncbi/MedCPT-Article-Encoder",
    "medcpt-cross": "ncbi/MedCPT-Cross-Encoder",
}
SYNTHESIS_REPO_ID = "UCSC-VLAA/MedReason-8B"
PREFERRED_TRM_CHECKPOINTS = (
    "medical_trm.pt",
    "medical_trm.ckpt",
    "medical_trm_finetuned.pt",
    "medical_trm_finetuned.ckpt",
    "minimed_trm_medical.pt",
    "minimed_trm_medical.ckpt",
    "sudoku_extreme.pt",
    "sudoku_extreme.ckpt",
    "sudoku-extreme.pt",
    "sudoku-extreme.ckpt",
)
REQUEST_TIMEOUT_SECONDS = 60
DEFAULT_GIT_DEPTH = "1"
KAGGLE_PIP_MODE_ENV = "MINIMED_KAGGLE_PIP_MODE"
KAGGLE_PIP_MODE_SAFE = "safe"
KAGGLE_PIP_MODE_FORCE = "force"
KAGGLE_RISKY_REQUIREMENT_NAMES = frozenset(
    {
        "spacy",
        "scispacy",
        "en-core-sci-lg",
        "llama-cpp-python",
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
    }
)


@dataclass(slots=True)
class SetupSummary:
    pip_installed: bool
    primekg_files: list[str]
    sapbert_path: str | None
    medcpt_paths: dict[str, str]
    synthesis_model_path: str | None
    trm_repo_path: str | None
    warnings: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up real MiniMed Prime integration dependencies and checkpoints.")
    parser.add_argument("--skip-pip", action="store_true", help="Skip pip installation steps.")
    parser.add_argument("--skip-downloads", action="store_true", help="Skip PrimeKG/model downloads.")
    parser.add_argument("--dry-run", action="store_true", help="Log planned actions without changing the environment.")
    parser.add_argument("--kaggle", action="store_true", help="Enable Kaggle-oriented path and memory handling.")
    args = parser.parse_args()

    warnings: list[str] = []
    setup_logger = StructuredLogger("setup_environment", DEFAULT_LOG_DIR)
    if args.kaggle or KaggleEnv.is_kaggle():
        T4Hardening.setup_memory()
    _ensure_directories()

    if not args.skip_pip:
        install_requirements(REQUIREMENTS_PATH, dry_run=args.dry_run, warnings=warnings)

    primekg_files: list[str] = []
    sapbert_path: str | None = None
    medcpt_paths: dict[str, str] = {}
    synthesis_model_path: str | None = None
    trm_repo_path: str | None = materialize_attached_tiny_recursive_models(
        dry_run=args.dry_run,
        warnings=warnings,
    )

    if not args.skip_downloads:
        session = requests.Session()
        session.headers.update({"User-Agent": "MiniMedPrimeSetup/1.0"})
        primekg_files = download_primekg(session=session, target_dir=PRIMEKG_DIR, dry_run=args.dry_run, warnings=warnings)
        sapbert_path = download_hf_snapshot(
            repo_id=SAPBERT_REPO_ID,
            target_dir=SAPBERT_DIR,
            dry_run=args.dry_run,
            warnings=warnings,
        )
        medcpt_paths = download_medcpt(dry_run=args.dry_run, warnings=warnings)
        synthesis_model_path = download_hf_snapshot(
            repo_id=SYNTHESIS_REPO_ID,
            target_dir=SYNTHESIS_MODEL_DIR,
            dry_run=args.dry_run,
            warnings=warnings,
        )
        if trm_repo_path is None:
            trm_repo_path = clone_tiny_recursive_models(target_dir=TRM_REPO_DIR, dry_run=args.dry_run, warnings=warnings)
    else:
        primekg_files = _existing_primekg_files(PRIMEKG_DIR)
        sapbert_path = _existing_model_path(SAPBERT_DIR)
        medcpt_paths = _existing_medcpt_paths()
        synthesis_model_path = _existing_model_path(SYNTHESIS_MODEL_DIR)

    summary = SetupSummary(
        pip_installed=not args.skip_pip,
        primekg_files=primekg_files,
        sapbert_path=sapbert_path,
        medcpt_paths=medcpt_paths,
        synthesis_model_path=synthesis_model_path,
        trm_repo_path=trm_repo_path,
        warnings=warnings,
    )
    logger.info("Integration setup summary:\n{}", json.dumps(asdict(summary), indent=2, sort_keys=True))
    setup_logger.log_event(
        "setup_complete",
        {
            "pip_installed": not args.skip_pip,
            "primekg_files": primekg_files,
            "sapbert_path": sapbert_path,
            "medcpt_paths": medcpt_paths,
            "synthesis_model_path": synthesis_model_path,
            "trm_repo_path": trm_repo_path,
            "warnings": warnings,
            "backend_used": "kaggle" if args.kaggle or KaggleEnv.is_kaggle() else "local",
            "latency_ms": 0.0,
        },
    )
    return 0


def _ensure_directories() -> None:
    for path in (
        DATA_DIR,
        CHECKPOINT_DIR,
        KG_DIR,
        PRIMEKG_DIR,
        SAPBERT_DIR,
        MEDCPT_QUERY_DIR,
        MEDCPT_ARTICLE_DIR,
        MEDCPT_CROSS_DIR,
        SYNTHESIS_MODEL_DIR,
        EXTERNAL_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def install_requirements(requirements_path: Path, dry_run: bool, warnings: list[str]) -> None:
    if not requirements_path.exists():
        warnings.append(f"Requirements file missing: {requirements_path}")
        logger.warning(f"Requirements file missing: {requirements_path}")
        return

    if _kaggle_pip_mode() == KAGGLE_PIP_MODE_SAFE:
        logger.info(
            "Kaggle safe pip mode is active; preinstalled packages will be preserved. "
            f"Set {KAGGLE_PIP_MODE_ENV}={KAGGLE_PIP_MODE_FORCE} to force pip installs into the notebook image."
        )

    requirements = [line.strip() for line in requirements_path.read_text(encoding="utf-8").splitlines()]
    standard_requirements: list[str] = []
    pyg_requirements: list[str] = []
    llama_requirement: str | None = None

    for requirement in requirements:
        if not requirement or requirement.startswith("#"):
            continue
        lowered = requirement.lower()
        if lowered.startswith("torch-geometric") or lowered.startswith("torch-scatter") or lowered.startswith("torch-sparse"):
            pyg_requirements.append(requirement)
            continue
        if lowered.startswith("llama-cpp-python"):
            llama_requirement = requirement
            continue
        standard_requirements.append(requirement)

    for requirement in standard_requirements:
        _pip_install(requirement=requirement, dry_run=dry_run, warnings=warnings)

    install_pyg_requirements(requirements=pyg_requirements, dry_run=dry_run, warnings=warnings)
    if llama_requirement is not None:
        if _should_skip_llama_cpp_install():
            message = "Skipping llama-cpp-python because the active synthesis backend does not require GGUF inference."
            warnings.append(message)
            logger.info(message)
        else:
            install_llama_cpp(requirement=llama_requirement, dry_run=dry_run, warnings=warnings)


def install_pyg_requirements(requirements: Iterable[str], dry_run: bool, warnings: list[str]) -> None:
    requirements = list(requirements)
    if not requirements:
        return

    wheel_index = _resolve_pyg_wheel_index()
    extra_args = ["-f", wheel_index] if wheel_index is not None else []
    if wheel_index is None:
        warning = "Could not resolve a PyG wheel index from the current torch installation; attempting default pip install."
        warnings.append(warning)
        logger.warning(warning)

    for requirement in requirements:
        _pip_install(requirement=requirement, extra_args=extra_args, dry_run=dry_run, warnings=warnings)


def install_llama_cpp(requirement: str, dry_run: bool, warnings: list[str]) -> None:
    env = os.environ.copy()
    extra_args: list[str] = []
    if _has_nvidia_gpu():
        env.setdefault("CMAKE_ARGS", "-DGGML_CUDA=on")
        env.setdefault("FORCE_CMAKE", "1")
        extra_args.extend(["--force-reinstall", "--no-cache-dir"])
    _pip_install(requirement=requirement, extra_args=extra_args, env=env, dry_run=dry_run, warnings=warnings)


def _pip_install(
    requirement: str,
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
    warnings: list[str] | None = None,
) -> bool:
    should_skip, skip_reason = _should_skip_pip_install(requirement)
    if should_skip:
        logger.info(skip_reason)
        return True

    command = [sys.executable, "-m", "pip", "install", requirement]
    if extra_args:
        command.extend(extra_args)
    if dry_run:
        logger.info("Dry run: {}", " ".join(command))
        return True
    logger.info("Installing requirement: {}", requirement)
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return True
    warning = f"pip install failed for '{requirement}': {completed.stderr.strip() or completed.stdout.strip()}"
    if warnings is not None:
        warnings.append(warning)
    logger.warning(warning)
    return False


def download_primekg(
    session: requests.Session,
    target_dir: Path,
    dry_run: bool,
    warnings: list[str],
) -> list[str]:
    dataset_url = f"{DATAVERSE_BASE_URL}/api/datasets/:persistentId/"
    try:
        response = session.get(
            dataset_url,
            params={"persistentId": PRIMEKG_PERSISTENT_ID},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        warning = f"PrimeKG metadata download failed: {exc}"
        warnings.append(warning)
        logger.warning(warning)
        return []

    metadata_path = target_dir / "dataverse_metadata.json"
    if dry_run:
        logger.info("Dry run: would write PrimeKG metadata to {}", metadata_path)
    else:
        metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    files = payload.get("data", {}).get("latestVersion", {}).get("files", [])
    downloaded_files: list[str] = []
    for file_entry in files:
        data_file = file_entry.get("dataFile", {})
        file_id = data_file.get("id")
        filename = str(data_file.get("filename", "")).strip()
        if not file_id or not filename.lower().endswith(".csv"):
            continue

        destination = target_dir / filename
        downloaded_files.append(str(destination))
        if destination.exists():
            logger.info("PrimeKG file already present: {}", destination)
            continue

        if dry_run:
            logger.info("Dry run: would download PrimeKG file {}", filename)
            continue

        access_url = f"{DATAVERSE_BASE_URL}/api/access/datafile/{file_id}"
        try:
            _stream_download(session=session, url=access_url, destination=destination)
            logger.info("Downloaded PrimeKG file {}", destination)
        except Exception as exc:
            warning = f"PrimeKG file download failed for {filename}: {exc}"
            warnings.append(warning)
            logger.warning(warning)
    return downloaded_files


def download_hf_snapshot(
    repo_id: str,
    target_dir: Path,
    dry_run: bool,
    warnings: list[str],
) -> str | None:
    config_path = target_dir / "config.json"
    if config_path.exists():
        logger.info("HF snapshot already present for {} at {}", repo_id, target_dir)
        return str(target_dir)
    if dry_run:
        logger.info("Dry run: would download HF snapshot {} to {}", repo_id, target_dir)
        return str(target_dir)

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        logger.info("Downloaded HF snapshot {} to {}", repo_id, target_dir)
        return str(target_dir)
    except Exception as exc:
        warning = f"HF snapshot download failed for {repo_id}: {exc}"
        warnings.append(warning)
        logger.warning(warning)
        return None


def download_medcpt(dry_run: bool, warnings: list[str]) -> dict[str, str]:
    target_dirs = {
        "medcpt-query": MEDCPT_QUERY_DIR,
        "medcpt-article": MEDCPT_ARTICLE_DIR,
        "medcpt-cross": MEDCPT_CROSS_DIR,
    }
    resolved_paths: dict[str, str] = {}
    for component_name, repo_id in MEDCPT_REPO_IDS.items():
        target_dir = target_dirs[component_name]
        target_dir.mkdir(parents=True, exist_ok=True)
        resolved = download_hf_snapshot(repo_id=repo_id, target_dir=target_dir, dry_run=dry_run, warnings=warnings)
        if resolved is not None:
            resolved_paths[component_name] = resolved
        else:
            logger.warning("Skipping MedCPT component {} because the repo is unavailable or gated.", repo_id)
    return resolved_paths


def clone_tiny_recursive_models(target_dir: Path, dry_run: bool, warnings: list[str]) -> str | None:
    if target_dir.exists() and (target_dir / ".git").exists():
        logger.info("Samsung TinyRecursiveModels repo already present at {}", target_dir)
        return str(target_dir)
    if target_dir.exists() and any(target_dir.iterdir()):
        warning = f"Target directory exists but is not a git repo: {target_dir}"
        warnings.append(warning)
        logger.warning(warning)
        return str(target_dir)
    if shutil.which("git") is None:
        warning = "git is not available; Samsung TinyRecursiveModels cannot be cloned automatically."
        warnings.append(warning)
        logger.warning(warning)
        return None

    command = [
        "git",
        "clone",
        "--depth",
        DEFAULT_GIT_DEPTH,
        SAMSUNG_TRM_REPO_URL,
        str(target_dir),
    ]
    if dry_run:
        logger.info("Dry run: {}", " ".join(command))
        return str(target_dir)

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        logger.info("Cloned Samsung TinyRecursiveModels into {}", target_dir)
        return str(target_dir)

    warning = f"Failed to clone Samsung TinyRecursiveModels: {completed.stderr.strip() or completed.stdout.strip()}"
    warnings.append(warning)
    logger.warning(warning)
    return None


def materialize_attached_tiny_recursive_models(dry_run: bool, warnings: list[str]) -> str | None:
    existing_repo = _find_tiny_recursive_models_repo(TRM_REPO_DIR)
    if existing_repo is not None:
        logger.info("TinyRecursiveModels repo already available at {}", existing_repo)
        return str(existing_repo)

    attached_repo = _find_tiny_recursive_models_repo(KaggleEnv.path("external/TinyRecursiveModels"))
    if attached_repo is not None:
        if not KaggleEnv.is_kaggle():
            return str(attached_repo)
        if dry_run:
            logger.info("Dry run: would copy attached TRM repo {} to {}", attached_repo, TRM_REPO_DIR)
            return str(TRM_REPO_DIR)
        TRM_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(attached_repo, TRM_REPO_DIR, dirs_exist_ok=True)
        logger.info("Copied attached TinyRecursiveModels repo from {} to {}", attached_repo, TRM_REPO_DIR)
        return str(TRM_REPO_DIR)

    archive_path = _find_tiny_recursive_models_archive()
    if archive_path is None:
        return None
    if dry_run:
        logger.info("Dry run: would extract TRM archive {} into {}", archive_path, TRM_REPO_DIR.parent)
        return str(TRM_REPO_DIR)

    TRM_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    try:
        if archive_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(TRM_REPO_DIR.parent)
        else:
            with tarfile.open(archive_path, mode="r:*") as archive:
                archive.extractall(TRM_REPO_DIR.parent)
    except Exception as exc:
        warning = f"Failed to extract TRM archive {archive_path}: {exc}"
        warnings.append(warning)
        logger.warning(warning)
        return None

    extracted_repo = _find_tiny_recursive_models_repo(TRM_REPO_DIR.parent)
    if extracted_repo is None:
        warning = f"TRM archive extracted but TinyRecursiveModels repo was not found under {TRM_REPO_DIR.parent}"
        warnings.append(warning)
        logger.warning(warning)
        return None
    logger.info("Extracted TinyRecursiveModels repo from {} to {}", archive_path, extracted_repo)
    return str(extracted_repo)


def _find_tiny_recursive_models_archive() -> Path | None:
    if not KaggleEnv.is_kaggle():
        return None
    dataset_root = KaggleEnv.INPUT_ROOT / TRM_KAGGLE_DATASET_SLUG
    if not dataset_root.exists():
        return None
    for archive_name in TRM_ARCHIVE_NAMES:
        candidate = dataset_root / archive_name
        if candidate.exists():
            return candidate
    for pattern in ("*.tar", "*.zip"):
        try:
            candidate = next(dataset_root.glob(f"**/{pattern}"))
        except (StopIteration, OSError):
            continue
        return candidate
    return None


def _find_tiny_recursive_models_repo(root: Path) -> Path | None:
    if not root.exists():
        return None
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


def _stream_download(session: requests.Session, url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _resolve_pyg_wheel_index() -> str | None:
    try:
        import torch
    except Exception:
        return None

    base_version = str(torch.__version__).split("+", maxsplit=1)[0]
    cuda_version = getattr(torch.version, "cuda", None)
    if cuda_version:
        cuda_tag = "cu" + cuda_version.replace(".", "")
        return f"https://data.pyg.org/whl/torch-{base_version}+{cuda_tag}.html"
    return f"https://data.pyg.org/whl/torch-{base_version}+cpu.html"


def _has_nvidia_gpu() -> bool:
    if shutil.which("nvidia-smi") is not None:
        return True
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def _existing_primekg_files(path: Path) -> list[str]:
    candidate_names = ("edges.csv", "nodes.csv", "kg.csv", "kg_raw.csv")
    if path.is_file():
        return [str(path)]
    if not path.exists():
        return []
    return [str(path / name) for name in candidate_names if (path / name).exists()]


def _existing_model_path(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_file():
        return str(path)
    if path.is_dir() and any(path.iterdir()):
        return str(path)
    return None


def _existing_medcpt_paths() -> dict[str, str]:
    target_dirs = {
        "medcpt-query": MEDCPT_QUERY_DIR,
        "medcpt-article": MEDCPT_ARTICLE_DIR,
        "medcpt-cross": MEDCPT_CROSS_DIR,
    }
    resolved_paths: dict[str, str] = {}
    for component_name, target_dir in target_dirs.items():
        existing = _existing_model_path(target_dir)
        if existing is not None:
            resolved_paths[component_name] = existing
    return resolved_paths


def _kaggle_pip_mode() -> str:
    if not KaggleEnv.is_kaggle():
        return "disabled"
    raw_mode = (os.getenv(KAGGLE_PIP_MODE_ENV) or KAGGLE_PIP_MODE_SAFE).strip().lower()
    return KAGGLE_PIP_MODE_FORCE if raw_mode == KAGGLE_PIP_MODE_FORCE else KAGGLE_PIP_MODE_SAFE


def _should_skip_pip_install(requirement: str) -> tuple[bool, str]:
    parsed_requirement = _parse_requirement(requirement)
    installed_version = _installed_requirement_version(parsed_requirement)
    if parsed_requirement is not None and installed_version is not None and _requirement_version_satisfies(
        parsed_requirement,
        installed_version,
    ):
        return True, f"Requirement already satisfied; skipping pip install for '{requirement}'."

    if _kaggle_pip_mode() != KAGGLE_PIP_MODE_SAFE:
        return False, ""

    if parsed_requirement is not None and installed_version is not None:
        return (
            True,
            "Kaggle safe pip mode is preserving the base image package "
            f"'{parsed_requirement.name}=={installed_version}' instead of mutating it for '{requirement}'. "
            f"Set {KAGGLE_PIP_MODE_ENV}={KAGGLE_PIP_MODE_FORCE} to override.",
        )

    if _is_kaggle_risky_requirement(parsed_requirement, requirement):
        return (
            True,
            "Kaggle safe pip mode is skipping risky native requirement "
            f"'{requirement}'. Set {KAGGLE_PIP_MODE_ENV}={KAGGLE_PIP_MODE_FORCE} to override.",
        )

    return False, ""


def _parse_requirement(requirement: str) -> Any | None:
    if Requirement is None:
        return None
    try:
        return Requirement(requirement)
    except Exception:
        return None


def _installed_requirement_version(parsed_requirement: Any | None) -> str | None:
    if parsed_requirement is None:
        return None
    distribution_name = getattr(parsed_requirement, "name", None)
    if not distribution_name:
        return None
    candidate_names = [str(distribution_name)]
    hyphenated_name = str(distribution_name).replace("_", "-")
    underscored_name = str(distribution_name).replace("-", "_")
    for candidate_name in (hyphenated_name, underscored_name):
        if candidate_name not in candidate_names:
            candidate_names.append(candidate_name)
    for candidate_name in candidate_names:
        try:
            return importlib_metadata.version(candidate_name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return None


def _requirement_version_satisfies(parsed_requirement: Any, installed_version: str) -> bool:
    specifier = getattr(parsed_requirement, "specifier", None)
    if specifier is None:
        return True
    try:
        return bool(specifier.contains(installed_version, prereleases=True))
    except Exception:
        return False


def _canonical_requirement_name(name: str) -> str:
    if canonicalize_name is not None:
        return str(canonicalize_name(name))
    return str(name).strip().lower().replace("_", "-")


def _is_kaggle_risky_requirement(parsed_requirement: Any | None, requirement: str) -> bool:
    if parsed_requirement is not None:
        requirement_name = getattr(parsed_requirement, "name", None)
        if requirement_name and _canonical_requirement_name(str(requirement_name)) in KAGGLE_RISKY_REQUIREMENT_NAMES:
            return True
        if getattr(parsed_requirement, "url", None):
            normalized_name = _canonical_requirement_name(str(requirement_name or ""))
            if normalized_name.startswith("en-core-sci"):
                return True
    lowered_requirement = requirement.lower()
    if lowered_requirement.startswith(("http://", "https://")) and "scispacy" in lowered_requirement:
        return True
    return False


def _should_skip_llama_cpp_install() -> bool:
    synthesis_hint = (os.getenv("MINIMED_SYNTHESIS_MODEL") or os.getenv("MINIMED_SYNTHESIS_MODEL_PATH") or "").strip().lower()
    if synthesis_hint.startswith(("gpt-", "o1", "o3")):
        return True
    if os.getenv("OPENAI_API_KEY"):
        medreason_dir = _resolve_asset_dir("data/checkpoints/medreason-8b")
        if medreason_dir.exists():
            return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
