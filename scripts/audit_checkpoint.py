#!/usr/bin/env python3
"""
Deep audit TRM checkpoint vs code compatibility.

Run:
  python scripts/audit_checkpoint.py --checkpoint path.pt --repo-root path/
"""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.util
import inspect
import json
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime in minimal envs.
    yaml = None  # type: ignore[assignment]


STATE_DICT_CANDIDATE_KEYS = ("ema_state_dict", "ema", "model_state_dict", "state_dict", "model")
KEY_PREFIXES_TO_STRIP = ("_orig_mod.", "module.", "model.", "ema_model.", "net.", "trm.")
DEFAULT_CONFIG: dict[str, Any] = {
    "batch_size": 1,
    "seq_len": 256,
    "puzzle_emb_ndim": 512,
    "num_puzzle_identifiers": 1,
    "vocab_size": 8192,
    "H_cycles": 3,
    "L_cycles": 4,
    "H_layers": 0,
    "L_layers": 2,
    "hidden_size": 512,
    "expansion": 4,
    "num_heads": 8,
    "pos_encodings": "rope",
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "halt_max_steps": 16,
    "halt_exploration_prob": 0.1,
    "forward_dtype": "bfloat16",
    "mlp_t": False,
    "puzzle_emb_len": 16,
    "no_ACT_continue": True,
}
MODEL_CONFIG_KEYS = set(DEFAULT_CONFIG)


@contextmanager
def temporary_sys_paths(*paths: Path):
    inserted: list[str] = []
    for path in paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
            inserted.append(path_str)
    try:
        importlib.invalidate_caches()
        yield
    finally:
        for path_str in inserted:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": int(value.numel()),
        }
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value


def load_checkpoint(checkpoint_path: Path) -> Any:
    """Load checkpoint on CPU. mmap is best-effort for large raw state dicts."""
    load_attempts = (
        {"map_location": "cpu", "mmap": True, "weights_only": False},
        {"map_location": "cpu", "weights_only": False},
        {"map_location": "cpu"},
    )
    last_error: Exception | None = None
    for kwargs in load_attempts:
        try:
            return torch.load(checkpoint_path, **kwargs)
        except TypeError as exc:
            last_error = exc
            continue
        except RuntimeError as exc:
            last_error = exc
            continue
    assert last_error is not None
    raise last_error


def is_tensor_dict(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(isinstance(item, torch.Tensor) for item in value.values())


def extract_state_dict(checkpoint: Any) -> tuple[dict[str, torch.Tensor], str]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(checkpoint).__name__}")

    for key in STATE_DICT_CANDIDATE_KEYS:
        value = checkpoint.get(key)
        if is_tensor_dict(value):
            return value, key

    if is_tensor_dict(checkpoint):
        return checkpoint, "top_level_tensor_dict"

    tensor_values = {key: value for key, value in checkpoint.items() if isinstance(value, torch.Tensor)}
    if tensor_values:
        return tensor_values, "top_level_filtered_tensor_dict"

    nested_tensor_dicts = {
        key: value for key, value in checkpoint.items() if isinstance(value, dict) and is_tensor_dict(value)
    }
    if nested_tensor_dicts:
        first_key = next(iter(nested_tensor_dicts))
        return nested_tensor_dicts[first_key], f"nested:{first_key}"

    raise ValueError("Could not locate a tensor state_dict in checkpoint payload.")


def normalize_key(key: str) -> str:
    normalized = key
    changed = True
    while changed:
        changed = False
        for prefix in KEY_PREFIXES_TO_STRIP:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
    return normalized


def normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized[normalize_key(key)] = value
    return normalized


def tensor_shape(value: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value.shape)


def state_sample(state_dict: dict[str, torch.Tensor], limit: int = 20) -> list[dict[str, Any]]:
    sample: list[dict[str, Any]] = []
    for key, value in list(state_dict.items())[:limit]:
        sample.append(
            {
                "key": key,
                "normalized_key": normalize_key(key),
                "shape": list(tensor_shape(value)),
                "dtype": str(value.dtype),
                "numel": int(value.numel()),
            }
        )
    return sample


def state_summary(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    total_numel = 0
    total_bytes = 0
    dtype_counts: dict[str, int] = {}
    prefix_counts: dict[str, int] = {}
    for key, value in state_dict.items():
        total_numel += int(value.numel())
        total_bytes += int(value.numel() * value.element_size())
        dtype_counts[str(value.dtype)] = dtype_counts.get(str(value.dtype), 0) + 1
        prefix = key.split(".", 1)[0]
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    return {
        "tensor_count": len(state_dict),
        "total_numel": total_numel,
        "total_bytes": total_bytes,
        "dtypes": dtype_counts,
        "top_level_prefix_counts": dict(sorted(prefix_counts.items(), key=lambda item: item[1], reverse=True)[:20]),
    }


def compare_state_dicts(
    model_state: dict[str, torch.Tensor],
    checkpoint_state: dict[str, torch.Tensor],
    *,
    normalize_checkpoint_keys: bool,
) -> dict[str, Any]:
    checkpoint_by_key: dict[str, tuple[str, torch.Tensor]] = {}
    duplicate_normalized_keys: list[str] = []
    for raw_key, value in checkpoint_state.items():
        candidate_key = normalize_key(raw_key) if normalize_checkpoint_keys else raw_key
        if candidate_key in checkpoint_by_key:
            duplicate_normalized_keys.append(candidate_key)
            continue
        checkpoint_by_key[candidate_key] = (raw_key, value)

    model_keys = set(model_state)
    checkpoint_keys = set(checkpoint_by_key)
    matched_keys = sorted(model_keys & checkpoint_keys)
    missing_keys = sorted(model_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - model_keys)
    shape_mismatches: list[dict[str, Any]] = []
    compatible_keys: list[str] = []

    for key in matched_keys:
        raw_key, checkpoint_tensor = checkpoint_by_key[key]
        model_tensor = model_state[key]
        model_shape = tensor_shape(model_tensor)
        checkpoint_shape = tensor_shape(checkpoint_tensor)
        if model_shape == checkpoint_shape:
            compatible_keys.append(key)
        else:
            shape_mismatches.append(
                {
                    "key": key,
                    "raw_checkpoint_key": raw_key,
                    "model_shape": list(model_shape),
                    "checkpoint_shape": list(checkpoint_shape),
                }
            )

    compatible_count = len(compatible_keys)
    return {
        "model_tensor_count": len(model_state),
        "checkpoint_tensor_count": len(checkpoint_state),
        "checkpoint_key_count_after_normalization": len(checkpoint_by_key),
        "matched_key_count": len(matched_keys),
        "compatible_tensor_count": compatible_count,
        "missing_key_count": len(missing_keys),
        "unexpected_key_count": len(unexpected_keys),
        "shape_mismatch_count": len(shape_mismatches),
        "duplicate_normalized_key_count": len(duplicate_normalized_keys),
        "full_match": compatible_count == len(model_state) == len(checkpoint_by_key) and not shape_mismatches,
        "partial_match": compatible_count > 0 and not shape_mismatches,
        "missing_keys_sample": missing_keys[:10],
        "unexpected_keys_sample": unexpected_keys[:10],
        "compatible_keys_sample": compatible_keys[:10],
        "shape_mismatches_sample": shape_mismatches[:10],
        "duplicate_normalized_keys_sample": duplicate_normalized_keys[:10],
    }


def read_yaml_file(path: Path) -> dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def resolve_simple_interpolations(payload: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(payload)
    for key, value in list(resolved.items()):
        if isinstance(value, str) and value.startswith("${.") and value.endswith("}"):
            referenced_key = value[3:-1]
            if referenced_key in resolved:
                resolved[key] = resolved[referenced_key]
    return resolved


def extract_arch_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("arch"), dict):
        payload = dict(payload["arch"])
    return resolve_simple_interpolations({key: value for key, value in payload.items() if key not in {"name", "loss"}})


def extract_arch_identifier(payload: dict[str, Any]) -> str | None:
    if isinstance(payload.get("arch"), dict):
        value = payload["arch"].get("name")
    else:
        value = payload.get("name")
    return str(value) if value else None


def load_arch_identifiers(repo_root: Path, checkpoint_path: Path) -> dict[str, str | None]:
    sidecar_payload: dict[str, Any] = {}
    for name in ("all_config.yaml", "config.yaml"):
        candidate = checkpoint_path.parent / name
        if candidate.exists():
            sidecar_payload = read_yaml_file(candidate)
            break
    repo_payload = read_yaml_file(repo_root / "config" / "arch" / "trm.yaml")
    return {
        "sidecar": extract_arch_identifier(sidecar_payload),
        "repo_trm_yaml": extract_arch_identifier(repo_payload),
    }


def class_identifier(repo_root: Path, py_file: Path, class_name: str) -> str:
    models_root = repo_root / "models"
    try:
        relative = py_file.relative_to(models_root)
        module_path = ".".join(relative.with_suffix("").parts)
        return f"{module_path}@{class_name}"
    except ValueError:
        try:
            relative = py_file.relative_to(repo_root)
            module_path = ".".join(relative.with_suffix("").parts)
            return f"{module_path}@{class_name}"
        except ValueError:
            return f"{py_file.stem}@{class_name}"


def infer_config_from_state_dict(normalized_state: dict[str, torch.Tensor]) -> dict[str, Any]:
    inferred: dict[str, Any] = {}

    token_embedding = normalized_state.get("inner.embed_tokens.embedding_weight")
    if token_embedding is None:
        token_embedding = normalized_state.get("embed_tokens.embedding_weight")
    if isinstance(token_embedding, torch.Tensor) and token_embedding.ndim == 2:
        inferred["vocab_size"] = int(token_embedding.shape[0])
        inferred["hidden_size"] = int(token_embedding.shape[1])

    lm_head = normalized_state.get("inner.lm_head.weight")
    if lm_head is None:
        lm_head = normalized_state.get("lm_head.weight")
    if isinstance(lm_head, torch.Tensor) and lm_head.ndim == 2:
        inferred.setdefault("vocab_size", int(lm_head.shape[0]))
        inferred.setdefault("hidden_size", int(lm_head.shape[1]))

    q_head = normalized_state.get("inner.q_head.weight")
    if isinstance(q_head, torch.Tensor) and q_head.ndim == 2:
        inferred.setdefault("hidden_size", int(q_head.shape[1]))

    puzzle_embedding = normalized_state.get("inner.puzzle_emb.weights")
    if puzzle_embedding is None:
        puzzle_embedding = normalized_state.get("puzzle_emb.weights")
    if isinstance(puzzle_embedding, torch.Tensor) and puzzle_embedding.ndim == 2:
        inferred["num_puzzle_identifiers"] = int(puzzle_embedding.shape[0])
        inferred["puzzle_emb_ndim"] = int(puzzle_embedding.shape[1])

    max_l_layer_index = -1
    for key in normalized_state:
        marker = "L_level.layers."
        if marker not in key:
            continue
        suffix = key.split(marker, 1)[1]
        try:
            layer_index = int(suffix.split(".", 1)[0])
        except ValueError:
            continue
        max_l_layer_index = max(max_l_layer_index, layer_index)
    if max_l_layer_index >= 0:
        inferred["L_layers"] = max_l_layer_index + 1

    if "inner.H_init" in normalized_state or "H_init" in normalized_state:
        inferred.setdefault("has_H_init", True)
    if "inner.L_init" in normalized_state or "L_init" in normalized_state:
        inferred.setdefault("has_L_init", True)

    return inferred


def build_model_config(repo_root: Path, checkpoint_path: Path, normalized_state: dict[str, torch.Tensor]) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)

    repo_arch = extract_arch_payload(read_yaml_file(repo_root / "config" / "arch" / "trm.yaml"))
    sidecar_config: dict[str, Any] = {}
    for name in ("all_config.yaml", "config.yaml"):
        candidate = checkpoint_path.parent / name
        if candidate.exists():
            sidecar_config = extract_arch_payload(read_yaml_file(candidate))
            break

    inferred = infer_config_from_state_dict(normalized_state)
    config.update(repo_arch)
    config.update(sidecar_config)
    config.update({key: value for key, value in inferred.items() if key in MODEL_CONFIG_KEYS})
    config["batch_size"] = int(config.get("batch_size") or 1)
    config["seq_len"] = int(config.get("seq_len") or DEFAULT_CONFIG["seq_len"])
    return {key: config[key] for key in MODEL_CONFIG_KEYS if key in config}


def config_class_for_module(module: Any, class_name: str) -> type[Any] | None:
    candidate_names = [
        f"{class_name}Config",
        f"{class_name.replace('_Inner', '')}Config",
        f"{class_name.replace('Block', '')}Config",
        f"{class_name.replace('ReasoningModule', '')}Config",
    ]
    if class_name.startswith("TinyRecursiveReasoningModel_ACTV1"):
        candidate_names.append("TinyRecursiveReasoningModel_ACTV1Config")
    if class_name.startswith("HierarchicalReasoningModel_ACTV1"):
        candidate_names.append("HierarchicalReasoningModel_ACTV1Config")
    if class_name.startswith("Model_ACTV2"):
        candidate_names.append("Model_ACTV2Config")

    for candidate_name in candidate_names:
        candidate = getattr(module, candidate_name, None)
        if inspect.isclass(candidate):
            return candidate
    return None


def instantiate_module_candidate(obj: type[nn.Module], module: Any, config_dict: dict[str, Any]) -> tuple[nn.Module, str]:
    attempts: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = [
        ("no_args", (), {}),
        ("config_dict", (config_dict,), {}),
    ]
    config_cls = config_class_for_module(module, obj.__name__)
    if config_cls is not None:
        try:
            config_obj = config_cls(**config_dict)
            attempts.append((f"{config_cls.__name__}_object", (config_obj,), {}))
        except Exception:
            pass

    last_error: Exception | None = None
    for strategy, args, kwargs in attempts:
        try:
            with torch.device("meta"):
                return obj(*args, **kwargs), strategy
        except Exception as exc:
            last_error = exc
            continue
    assert last_error is not None
    raise last_error


def import_module_from_file(py_file: Path, repo_root: Path) -> Any:
    module_name = f"audit_{abs(hash(str(py_file.resolve())))}_{py_file.stem}"
    spec = importlib.util.spec_from_file_location(module_name, py_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {py_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    with temporary_sys_paths(repo_root, repo_root / "models"):
        spec.loader.exec_module(module)
    return module


def try_tiny_recursive_model(repo_root: Path, state_dict: dict[str, torch.Tensor], config_dict: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "module_candidates": ["recursive_reasoning.trm", "models.recursive_reasoning.trm"],
        "class": "TinyRecursiveModel",
        "found": False,
        "error": None,
    }
    with temporary_sys_paths(repo_root, repo_root / "models"):
        for module_name in result["module_candidates"]:
            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, "TinyRecursiveModel")
                result["found"] = True
                model, init_strategy = instantiate_module_candidate(cls, module, config_dict)
                model_state = model.state_dict()
                result["init_strategy"] = init_strategy
                result["exact"] = compare_state_dicts(model_state, state_dict, normalize_checkpoint_keys=False)
                result["prefix_normalized"] = compare_state_dicts(model_state, state_dict, normalize_checkpoint_keys=True)
                return result
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def scan_model_classes(
    repo_root: Path,
    state_dict: dict[str, torch.Tensor],
    config_dict: dict[str, Any],
    *,
    max_candidates: int,
    preferred_identifier: str | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    py_files = sorted(path for path in repo_root.rglob("*.py") if "__pycache__" not in path.parts)

    for py_file in py_files:
        try:
            module = import_module_from_file(py_file, repo_root)
        except Exception:
            continue

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            if not issubclass(obj, nn.Module) or obj is nn.Module:
                continue

            try:
                instance, init_strategy = instantiate_module_candidate(obj, module, config_dict)
                model_state = instance.state_dict()
                exact = compare_state_dicts(model_state, state_dict, normalize_checkpoint_keys=False)
                normalized = compare_state_dicts(model_state, state_dict, normalize_checkpoint_keys=True)
                best_count = max(exact["compatible_tensor_count"], normalized["compatible_tensor_count"])
                if best_count > 0:
                    identifier = class_identifier(repo_root, py_file, name)
                    candidates.append(
                        {
                            "file": str(py_file),
                            "class": name,
                            "identifier": identifier,
                            "matches_preferred_arch": identifier == preferred_identifier,
                            "init_strategy": init_strategy,
                            "matched": best_count,
                            "exact": exact,
                            "prefix_normalized": normalized,
                        }
                    )
                del instance
                gc.collect()
            except Exception:
                continue

    candidates.sort(
        key=lambda item: (
            1 if item.get("matches_preferred_arch") else 0,
            item["prefix_normalized"]["compatible_tensor_count"],
            item["exact"]["compatible_tensor_count"],
            -item["prefix_normalized"]["missing_key_count"],
        ),
        reverse=True,
    )
    return candidates[:max_candidates]


def verify_config_compatibility(
    repo_root: Path,
    checkpoint_path: Path,
    normalized_state: dict[str, torch.Tensor],
    best_candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    inferred = infer_config_from_state_dict(normalized_state)
    repo_config_path = repo_root / "config" / "arch" / "trm.yaml"
    sidecar_config_path = checkpoint_path.parent / "all_config.yaml"
    repo_arch = extract_arch_payload(read_yaml_file(repo_config_path))
    sidecar_arch = extract_arch_payload(read_yaml_file(sidecar_config_path))

    compared_fields = [
        "hidden_size",
        "puzzle_emb_ndim",
        "L_layers",
        "vocab_size",
        "num_puzzle_identifiers",
        "pos_encodings",
        "puzzle_emb_len",
    ]

    def compare_config(name: str, payload: dict[str, Any]) -> dict[str, Any]:
        mismatches: dict[str, Any] = {}
        missing_dynamic_fields: list[str] = []
        for field in compared_fields:
            if field not in inferred:
                continue
            if field not in payload:
                missing_dynamic_fields.append(field)
                continue
            if payload[field] != inferred[field]:
                mismatches[field] = {"config": payload[field], "checkpoint_inferred": inferred[field]}
        return {
            "name": name,
            "exists": bool(payload),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "missing_dynamic_fields": missing_dynamic_fields,
            "structure_matches_inferred_state": len(mismatches) == 0,
        }

    sidecar_vs_repo_differences: dict[str, Any] = {}
    for key in sorted(set(repo_arch) | set(sidecar_arch)):
        if repo_arch.get(key) != sidecar_arch.get(key):
            sidecar_vs_repo_differences[key] = {
                "repo_trm_yaml": repo_arch.get(key),
                "sidecar_all_config": sidecar_arch.get(key),
            }

    best_full_match = bool(best_candidate and best_candidate["prefix_normalized"]["full_match"])
    needs_sidecar_config = bool(sidecar_vs_repo_differences)
    seq_len_note = (
        "seq_len is not inferable from this checkpoint when RoPE buffers are non-persistent; "
        "use dataset metadata or the training config if exact ARC inference length matters."
    )

    return {
        "repo_config_path": str(repo_config_path),
        "sidecar_config_path": str(sidecar_config_path),
        "yaml_available": yaml is not None,
        "inferred_from_checkpoint": inferred,
        "repo_trm_yaml": compare_config("repo_config/arch/trm.yaml", repo_arch),
        "sidecar_all_config": compare_config("checkpoint_sidecar/all_config.yaml", sidecar_arch),
        "sidecar_vs_repo_differences": sidecar_vs_repo_differences,
        "best_candidate_prefix_normalized_full_match": best_full_match,
        "needs_sidecar_config_for_training_semantics": needs_sidecar_config,
        "notes": [seq_len_note],
    }


def audit_checkpoint(checkpoint_path: str, repo_root: str, max_candidates: int = 20) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    root = Path(repo_root).expanduser().resolve()

    print(f"=== AUDIT: {checkpoint} ===\n")
    print(f"Repo root: {root}")

    raw_checkpoint = load_checkpoint(checkpoint)
    top_level_keys = list(raw_checkpoint.keys()) if isinstance(raw_checkpoint, dict) else []
    print(f"Checkpoint payload type: {type(raw_checkpoint).__name__}")
    print(f"Checkpoint keys: {top_level_keys[:30]}")

    state_dict, state_source = extract_state_dict(raw_checkpoint)
    normalized_state = normalize_state_dict_keys(state_dict)
    arch_identifiers = load_arch_identifiers(root, checkpoint)
    preferred_arch_identifier = arch_identifiers.get("sidecar") or arch_identifiers.get("repo_trm_yaml")
    print(f"\nState dict source: {state_source}")
    print(f"State dict has {len(state_dict)} tensors")
    print("Sample keys (first 20):")
    for item in state_sample(state_dict, limit=20):
        print(f"  {item['key']}: {tuple(item['shape'])} dtype={item['dtype']} -> {item['normalized_key']}")

    config_dict = build_model_config(root, checkpoint, normalized_state)
    print("\nModel config used for instantiation:")
    print(json.dumps(json_safe(config_dict), indent=2, sort_keys=True))
    print(f"Preferred arch identifier: {preferred_arch_identifier}")

    tiny_recursive_result = try_tiny_recursive_model(root, state_dict, config_dict)
    if tiny_recursive_result.get("found"):
        print("\nTinyRecursiveModel was found and compared.")
    else:
        print(f"\nTinyRecursiveModel not usable: {tiny_recursive_result.get('error')}")

    print("\nScanning nn.Module classes in repo...")
    candidates = scan_model_classes(
        root,
        state_dict,
        config_dict,
        max_candidates=max_candidates,
        preferred_identifier=preferred_arch_identifier,
    )
    print(f"Found {len(candidates)} candidate class(es) with at least one compatible tensor.")
    for candidate in candidates[:10]:
        normalized = candidate["prefix_normalized"]
        exact = candidate["exact"]
        print(
            "  "
            f"{candidate['class']} [{Path(candidate['file']).relative_to(root)}] "
            f"identifier={candidate['identifier']} "
            f"exact={exact['compatible_tensor_count']} "
            f"normalized={normalized['compatible_tensor_count']} "
            f"full_match={normalized['full_match']} "
            f"preferred={candidate.get('matches_preferred_arch')}"
        )

    best_candidate = candidates[0] if candidates else None
    config_report = verify_config_compatibility(root, checkpoint, normalized_state, best_candidate)

    compatible = bool(best_candidate and best_candidate["prefix_normalized"]["partial_match"])
    full_match = bool(best_candidate and best_candidate["prefix_normalized"]["full_match"])
    conclusion: dict[str, Any] = {
        "compatible_after_prefix_normalization": compatible,
        "full_state_dict_match_after_prefix_normalization": full_match,
        "best_matching_class": None,
        "likely_root_cause": None,
        "checkpoint_corrupt": False,
        "needs_config_other_than_repo_trm_yaml": bool(config_report["needs_sidecar_config_for_training_semantics"]),
    }
    if best_candidate is not None:
        conclusion["best_matching_class"] = {
            "file": best_candidate["file"],
            "class": best_candidate["class"],
            "identifier": best_candidate["identifier"],
            "matches_preferred_arch": best_candidate.get("matches_preferred_arch", False),
            "init_strategy": best_candidate["init_strategy"],
            "prefix_normalized": best_candidate["prefix_normalized"],
        }
        if full_match:
            conclusion["likely_root_cause"] = (
                "Checkpoint is compatible with the official TRM class after stripping training prefixes "
                "such as _orig_mod. and model.; it is not compatible with fallback/internal wrapper key names. "
                "If multiple classes have identical state_dict structure, the checkpoint sidecar arch name selects the intended class."
            )
        else:
            conclusion["likely_root_cause"] = (
                "Checkpoint has partial matches only; inspect missing and shape mismatch counts for architecture drift."
            )
    else:
        conclusion["checkpoint_corrupt"] = len(state_dict) == 0
        conclusion["likely_root_cause"] = "No nn.Module class in the repo matched checkpoint tensor names and shapes."

    result = {
        "checkpoint_path": str(checkpoint),
        "repo_root": str(root),
        "payload": {
            "type": type(raw_checkpoint).__name__,
            "top_level_keys": top_level_keys[:30],
            "top_level_key_count": len(top_level_keys),
        },
        "state_dict_source": state_source,
        "state_summary": state_summary(state_dict),
        "state_sample": state_sample(state_dict, limit=20),
        "normalized_state_sample": state_sample(normalized_state, limit=20),
        "model_config_used": config_dict,
        "arch_identifiers": arch_identifiers,
        "preferred_arch_identifier": preferred_arch_identifier,
        "tiny_recursive_model": tiny_recursive_result,
        "candidates": candidates,
        "config_compatibility": config_report,
        "conclusion": conclusion,
    }
    if best_candidate is not None:
        best_counts = best_candidate["prefix_normalized"]
        result.update(
            {
                "compatible": compatible,
                "matched": best_counts["compatible_tensor_count"],
                "missing": best_counts["missing_key_count"],
                "unexpected": best_counts["unexpected_key_count"],
                "shape_mismatches": best_counts["shape_mismatch_count"],
                "model_class": best_candidate["identifier"],
            }
        )
    else:
        result.update(
            {
                "compatible": False,
                "matched": 0,
                "missing": None,
                "unexpected": len(state_dict),
                "shape_mismatches": None,
                "model_class": None,
            }
        )

    print("\n" + "=" * 50)
    print(f"COMPATIBILITY: {'COMPATIBLE' if compatible else 'INCOMPATIBLE'}")
    print(f"FULL MATCH AFTER PREFIX NORMALIZATION: {full_match}")
    if conclusion["best_matching_class"]:
        best = conclusion["best_matching_class"]
        print(f"BEST CLASS: {best['class']} ({best['file']})")
    print(f"LIKELY ROOT CAUSE: {conclusion['likely_root_cause']}")
    print("=" * 50)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Deep audit TRM checkpoint vs repo model definitions.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt/.pth/.ckpt checkpoint.")
    parser.add_argument("--repo-root", required=True, help="Path to TinyRecursiveModels repo root.")
    parser.add_argument("--max-candidates", type=int, default=20, help="Maximum matching classes to include in JSON.")
    args = parser.parse_args()

    try:
        result = audit_checkpoint(args.checkpoint, args.repo_root, max_candidates=args.max_candidates)
    except Exception as exc:
        traceback.print_exc()
        result = {
            "compatible": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        print(f"\nJSON_RESULT: {json.dumps(json_safe(result), sort_keys=True)}")
        return 1

    print(f"\nJSON_RESULT: {json.dumps(json_safe(result), sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
