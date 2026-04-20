from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.utils.kaggle_env import KaggleEnv

CODEBOOK_STATE_FILENAME = "codebook.pt"
CODEBOOK_META_FILENAME = "codebook_meta.json"
CODEBOOK_STATS_FILENAME = "codebook_stats.json"
TRAIN_PROGRESS_FILENAME = "train_progress.json"
TRAIN_LOG_FILENAME = "train_log.jsonl"


@dataclass(slots=True)
class CodebookArtifactPaths:
    output_dir: Path
    codebook_path: Path
    meta_path: Path
    stats_path: Path
    progress_path: Path
    log_path: Path


@dataclass(slots=True)
class FrozenCodebookArtifact:
    paths: CodebookArtifactPaths
    codebook_weight: torch.Tensor
    metadata: dict[str, Any]
    stats: dict[str, Any]
    progress: dict[str, Any]
    state: dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_schema_hash(values: Sequence[str]) -> str:
    payload = json.dumps([str(value) for value in values], ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_source_graph_signature(path: str | Path | None) -> str:
    if path is None:
        return "unavailable"
    resolved = path if isinstance(path, Path) and path.is_absolute() else KaggleEnv.path(Path(path))
    if not resolved.exists():
        return f"missing:{resolved.as_posix()}"
    try:
        file_payloads = [_file_signature_payload(candidate, root=resolved) for candidate in _signature_candidates(resolved)]
    except MemoryError:
        file_payloads = [_lightweight_file_signature_payload(candidate, root=resolved) for candidate in _signature_candidates(resolved)]
    except OSError:
        file_payloads = [_lightweight_file_signature_payload(candidate, root=resolved) for candidate in _signature_candidates(resolved)]
    payload = {
        "root": resolved.as_posix(),
        "is_dir": resolved.is_dir(),
        "files": file_payloads,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")).hexdigest()


def build_codebook_version(
    *,
    created_at: str,
    codebook_size: int,
    hidden_dim: int,
    seed: int,
    relation_schema_hash: str,
    node_type_schema_hash: str,
    source_graph_signature: str,
) -> str:
    compact_date = created_at.replace(":", "").replace("-", "").split(".")[0].replace("+0000", "Z").replace("+00:00", "Z")
    version_payload = {
        "codebook_size": int(codebook_size),
        "hidden_dim": int(hidden_dim),
        "seed": int(seed),
        "relation_schema_hash": relation_schema_hash,
        "node_type_schema_hash": node_type_schema_hash,
        "source_graph_signature": source_graph_signature,
    }
    version_hash = hashlib.sha256(
        json.dumps(version_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"vqcb-{compact_date}-{version_hash}"


def resolve_artifact_paths(path: str | Path) -> CodebookArtifactPaths:
    raw_path = Path(path)
    resolved_path = raw_path if raw_path.is_absolute() else KaggleEnv.path(raw_path)
    output_dir = resolved_path.parent if resolved_path.name == CODEBOOK_STATE_FILENAME else resolved_path
    return CodebookArtifactPaths(
        output_dir=output_dir,
        codebook_path=output_dir / CODEBOOK_STATE_FILENAME,
        meta_path=output_dir / CODEBOOK_META_FILENAME,
        stats_path=output_dir / CODEBOOK_STATS_FILENAME,
        progress_path=output_dir / TRAIN_PROGRESS_FILENAME,
        log_path=output_dir / TRAIN_LOG_FILENAME,
    )


def save_codebook_artifact(
    output_dir: str | Path,
    *,
    codebook_weight: torch.Tensor,
    metadata: Mapping[str, Any],
    stats: Mapping[str, Any],
    progress: Mapping[str, Any],
    extra_state: Mapping[str, Any] | None = None,
) -> CodebookArtifactPaths:
    paths = resolve_artifact_paths(output_dir)
    write_dir = KaggleEnv.ensure_writeable(paths.output_dir)
    paths = resolve_artifact_paths(write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "codebook_weight": codebook_weight.detach().cpu().to(dtype=torch.float32),
        **dict(extra_state or {}),
    }
    torch.save(state, paths.codebook_path)
    _write_json(paths.meta_path, dict(metadata))
    _write_json(paths.stats_path, dict(stats))
    _write_json(paths.progress_path, dict(progress))
    return paths


def load_frozen_codebook(path: str | Path) -> FrozenCodebookArtifact:
    paths = resolve_artifact_paths(path)
    if not paths.codebook_path.exists():
        raise FileNotFoundError(
            f"Frozen codebook artifact not found at {paths.codebook_path}. "
            f"Expected a directory containing {CODEBOOK_STATE_FILENAME} and JSON metadata files."
        )
    if not paths.meta_path.exists():
        raise FileNotFoundError(f"Frozen codebook metadata not found at {paths.meta_path}.")
    state = torch.load(paths.codebook_path, map_location="cpu")
    if not isinstance(state, Mapping):
        raise ValueError(f"Codebook artifact at {paths.codebook_path} must contain a mapping payload.")
    codebook_weight = state.get("codebook_weight")
    if not isinstance(codebook_weight, torch.Tensor):
        raise ValueError(f"Codebook artifact at {paths.codebook_path} is missing tensor field 'codebook_weight'.")
    metadata = _read_json(paths.meta_path)
    stats = _read_json(paths.stats_path, default={})
    progress = _read_json(paths.progress_path, default={})
    return FrozenCodebookArtifact(
        paths=paths,
        codebook_weight=codebook_weight.detach().cpu().to(dtype=torch.float32),
        metadata=metadata,
        stats=stats,
        progress=progress,
        state=dict(state),
    )


def assert_metadata_compatibility(
    metadata: Mapping[str, Any],
    *,
    expected_codebook_size: int,
    expected_hidden_dim: int,
    expected_relation_schema_hash: str,
    expected_node_type_schema_hash: str,
    expected_source_graph_signature: str | None = None,
    strict_source_graph_signature: bool = False,
) -> None:
    mismatches: list[str] = []
    if int(metadata.get("codebook_size", -1)) != int(expected_codebook_size):
        mismatches.append(
            f"codebook_size mismatch: artifact={metadata.get('codebook_size')} runtime={int(expected_codebook_size)}"
        )
    if int(metadata.get("hidden_dim", -1)) != int(expected_hidden_dim):
        mismatches.append(f"hidden_dim mismatch: artifact={metadata.get('hidden_dim')} runtime={int(expected_hidden_dim)}")
    if str(metadata.get("relation_schema_hash")) != str(expected_relation_schema_hash):
        mismatches.append(
            "relation_schema_hash mismatch: "
            f"artifact={metadata.get('relation_schema_hash')} runtime={expected_relation_schema_hash}"
        )
    if str(metadata.get("node_type_schema_hash")) != str(expected_node_type_schema_hash):
        mismatches.append(
            "node_type_schema_hash mismatch: "
            f"artifact={metadata.get('node_type_schema_hash')} runtime={expected_node_type_schema_hash}"
        )
    artifact_source_signature = metadata.get("source_graph_signature")
    if strict_source_graph_signature and expected_source_graph_signature is not None:
        if str(artifact_source_signature) != str(expected_source_graph_signature):
            mismatches.append(
                "source_graph_signature mismatch: "
                f"artifact={artifact_source_signature} runtime={expected_source_graph_signature}"
            )
    if mismatches:
        raise ValueError(
            "Frozen codebook metadata is incompatible with the current runtime:\n- " + "\n- ".join(mismatches)
        )


def apply_frozen_codebook(quantizer: Any, artifact: FrozenCodebookArtifact, *, freeze: bool = True) -> None:
    codebook = getattr(quantizer, "codebook", None)
    if codebook is None or not hasattr(codebook, "weight"):
        raise ValueError("Quantizer does not expose a loadable 'codebook.weight' parameter.")
    if tuple(codebook.weight.shape) != tuple(artifact.codebook_weight.shape):
        raise ValueError(
            "Frozen codebook weight shape mismatch: "
            f"artifact={tuple(artifact.codebook_weight.shape)} runtime={tuple(codebook.weight.shape)}"
        )
    with torch.no_grad():
        codebook.weight.copy_(artifact.codebook_weight.to(device=codebook.weight.device, dtype=codebook.weight.dtype))
    codebook.weight.requires_grad_(not freeze)


def quantize_with_codebook(sample_vectors: torch.Tensor, codebook_weight: torch.Tensor) -> torch.Tensor:
    if sample_vectors.ndim != 2:
        raise ValueError(f"Expected [num_vectors, hidden_dim] sample vectors, received shape {tuple(sample_vectors.shape)}.")
    if sample_vectors.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=sample_vectors.device)
    distances = torch.cdist(sample_vectors.to(dtype=torch.float32), codebook_weight.to(device=sample_vectors.device, dtype=torch.float32))
    return torch.argmin(distances, dim=-1).to(dtype=torch.long)


def inspect_codebook_usage(
    sample_vectors: torch.Tensor,
    codebook_weight: torch.Tensor,
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    token_ids = quantize_with_codebook(sample_vectors, codebook_weight)
    usage_counts = torch.bincount(token_ids, minlength=codebook_weight.size(0)) if token_ids.numel() else torch.zeros(
        (codebook_weight.size(0),), dtype=torch.long
    )
    total = int(token_ids.numel())
    dead_mask = usage_counts == 0
    dead_count = int(dead_mask.sum().item())
    top_values, top_indices = torch.topk(usage_counts, k=min(top_k, usage_counts.numel())) if usage_counts.numel() else (
        torch.zeros((0,), dtype=torch.long),
        torch.zeros((0,), dtype=torch.long),
    )
    return {
        "total_vectors": total,
        "unique_codes_used": int((usage_counts > 0).sum().item()),
        "dead_code_count": dead_count,
        "dead_code_ratio": float(dead_count / usage_counts.numel()) if usage_counts.numel() else 0.0,
        "usage_counts": usage_counts.tolist(),
        "top_used_clusters": [
            {
                "cluster_id": int(index.item()),
                "count": int(value.item()),
                "share": float(value.item() / total) if total else 0.0,
            }
            for value, index in zip(top_values, top_indices)
            if int(value.item()) > 0
        ],
        "unused_clusters": [int(index) for index in torch.nonzero(dead_mask, as_tuple=False).flatten().tolist()],
        "token_ids_preview": token_ids[: min(32, total)].detach().cpu().tolist(),
    }


def report_codebook_usage(usage_stats: Mapping[str, Any], *, top_k: int = 10) -> dict[str, Any]:
    top_used = list(usage_stats.get("top_used_clusters", []))[:top_k]
    unused = list(usage_stats.get("unused_clusters", []))[:top_k]
    return {
        "unique_codes_used": int(usage_stats.get("unique_codes_used", 0)),
        "dead_code_ratio": float(usage_stats.get("dead_code_ratio", 0.0)),
        "top_used_clusters": top_used,
        "unused_clusters": unused,
    }


def estimate_token_stability(
    sample_vectors: torch.Tensor,
    codebook_weight: torch.Tensor,
    *,
    repeats: int = 2,
) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("repeats must be at least 2 to estimate token stability.")
    passes = [quantize_with_codebook(sample_vectors, codebook_weight).detach().cpu() for _ in range(repeats)]
    baseline = passes[0] if passes else torch.empty((0,), dtype=torch.long)
    identical = [bool(torch.equal(baseline, pass_ids)) for pass_ids in passes[1:]]
    stable_pairs = sum(1 for item in identical if item)
    return {
        "repeats": int(repeats),
        "stable_pairs": int(stable_pairs),
        "pairwise_identical_ratio": float(stable_pairs / max(repeats - 1, 1)),
        "token_ids_preview": baseline[: min(32, baseline.numel())].tolist(),
    }


def append_train_log(path: str | Path, payload: Mapping[str, Any]) -> None:
    log_path = resolve_artifact_paths(path).log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=True, sort_keys=True) + "\n")


def _read_json(path: Path, *, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return dict(default or {})
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}, received {type(payload).__name__}.")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def _signature_candidates(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    preferred = [path / "edges.csv", path / "nodes.csv", path / "kg.csv"]
    candidates = [candidate for candidate in preferred if candidate.exists() and candidate.is_file()]
    if candidates:
        return candidates
    return sorted(candidate for candidate in path.glob("*.csv") if candidate.is_file())[:8]


def _file_signature_payload(path: Path, *, root: Path) -> dict[str, Any]:
    stat = path.stat()
    relative_path = path.name if root.is_file() else path.relative_to(root).as_posix()
    return {
        "path": relative_path,
        "size": int(stat.st_size),
        "sample_sha256": _sample_file_digest(path),
    }


def _sample_file_digest(path: Path, *, max_bytes: int = 65_536, chunk_size: int = 8_192) -> str:
    digest = hashlib.sha256()
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        remaining = max_bytes
        while remaining > 0:
            chunk = handle.read(min(chunk_size, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
        if file_size > max_bytes:
            tail_offset = max(file_size - max_bytes, 0)
            handle.seek(tail_offset)
            remaining = max_bytes
            while remaining > 0:
                chunk = handle.read(min(chunk_size, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                remaining -= len(chunk)
    return digest.hexdigest()


def _lightweight_file_signature_payload(path: Path, *, root: Path) -> dict[str, Any]:
    stat = path.stat()
    relative_path = path.name if root.is_file() else path.relative_to(root).as_posix()
    return {
        "path": relative_path,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sample_sha256": "unavailable",
    }


__all__ = [
    "CODEBOOK_META_FILENAME",
    "CODEBOOK_STATE_FILENAME",
    "CODEBOOK_STATS_FILENAME",
    "TRAIN_LOG_FILENAME",
    "TRAIN_PROGRESS_FILENAME",
    "FrozenCodebookArtifact",
    "append_train_log",
    "apply_frozen_codebook",
    "assert_metadata_compatibility",
    "build_codebook_version",
    "compute_schema_hash",
    "compute_source_graph_signature",
    "estimate_token_stability",
    "inspect_codebook_usage",
    "load_frozen_codebook",
    "quantize_with_codebook",
    "report_codebook_usage",
    "resolve_artifact_paths",
    "save_codebook_artifact",
    "utc_now_iso",
]
