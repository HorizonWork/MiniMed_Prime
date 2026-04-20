#!/usr/bin/env python3
"""
Isolated Samsung TRM load/forward test.

This intentionally avoids src.models.trm_wrapper so wrapper bugs can be
separated from checkpoint/model-definition bugs.

Run:
  python scripts/test_trm_isolated.py --repo-root external/TinyRecursiveModels
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
import traceback
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import torch

try:
    import yaml
except ImportError:  # pragma: no cover - runtime diagnostic only.
    yaml = None  # type: ignore[assignment]


DEFAULT_KAGGLE_REPO = Path("/kaggle/working/external/TinyRecursiveModels")
DEFAULT_LOCAL_REPO = Path("external/TinyRecursiveModels")
DEFAULT_CHECKPOINT_NAME = "trm_arc_v1_public_step_518071.pt"
STATE_DICT_KEYS = ("ema_state_dict", "ema", "model_state_dict", "state_dict", "model")
PREFIXES = ("_orig_mod.", "module.", "model.", "ema_model.")
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
    "forward_dtype": "float16",
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
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def resolve_interpolations(payload: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(payload)
    for key, value in list(resolved.items()):
        if isinstance(value, str) and value.startswith("${.") and value.endswith("}"):
            source_key = value[3:-1]
            if source_key in resolved:
                resolved[key] = resolved[source_key]
    return resolved


def arch_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("arch"), dict):
        payload = dict(payload["arch"])
    return resolve_interpolations({key: value for key, value in payload.items() if key not in {"name", "loss"}})


def arch_identifier(payload: dict[str, Any]) -> str | None:
    if isinstance(payload.get("arch"), dict):
        name = payload["arch"].get("name")
    else:
        name = payload.get("name")
    return str(name) if name else None


def load_checkpoint(path: Path) -> Any:
    attempts = (
        {"map_location": "cpu", "mmap": True, "weights_only": False},
        {"map_location": "cpu", "weights_only": False},
        {"map_location": "cpu"},
    )
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return torch.load(path, **kwargs)
        except (TypeError, RuntimeError) as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(payload).__name__}")
    for key in STATE_DICT_KEYS:
        value = payload.get(key)
        if isinstance(value, dict) and all(isinstance(item, torch.Tensor) for item in value.values()):
            return value
    if payload and all(isinstance(item, torch.Tensor) for item in payload.values()):
        return payload  # type: ignore[return-value]
    raise ValueError("No tensor state_dict found in checkpoint.")


def normalize_key(key: str) -> str:
    normalized = key
    changed = True
    while changed:
        changed = False
        for prefix in PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
    return normalized


def normalize_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {normalize_key(key): value for key, value in state.items()}


def infer_config(normalized_state: dict[str, torch.Tensor]) -> dict[str, Any]:
    inferred: dict[str, Any] = {}
    embed = normalized_state.get("inner.embed_tokens.embedding_weight")
    if isinstance(embed, torch.Tensor) and embed.ndim == 2:
        inferred["vocab_size"] = int(embed.shape[0])
        inferred["hidden_size"] = int(embed.shape[1])
    puzzle = normalized_state.get("inner.puzzle_emb.weights")
    if isinstance(puzzle, torch.Tensor) and puzzle.ndim == 2:
        inferred["num_puzzle_identifiers"] = int(puzzle.shape[0])
        inferred["puzzle_emb_ndim"] = int(puzzle.shape[1])
    max_layer = -1
    for key in normalized_state:
        prefix = "inner.L_level.layers."
        if not key.startswith(prefix):
            continue
        try:
            max_layer = max(max_layer, int(key[len(prefix) :].split(".", 1)[0]))
        except ValueError:
            pass
    if max_layer >= 0:
        inferred["L_layers"] = max_layer + 1
    return inferred


def build_config(repo_root: Path, checkpoint: Path, state: dict[str, torch.Tensor], seq_len: int, forward_dtype: str) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config.update(arch_payload(read_yaml(repo_root / "config" / "arch" / "trm.yaml")))
    for filename in ("all_config.yaml", "config.yaml"):
        sidecar = checkpoint.parent / filename
        if sidecar.exists():
            config.update(arch_payload(read_yaml(sidecar)))
            break
    config.update({key: value for key, value in infer_config(state).items() if key in MODEL_CONFIG_KEYS})
    config["batch_size"] = int(config.get("batch_size") or 1)
    config["seq_len"] = int(seq_len)
    config["forward_dtype"] = forward_dtype
    return {key: config[key] for key in MODEL_CONFIG_KEYS if key in config}


def import_model_class(repo_root: Path, identifier: str) -> type[torch.nn.Module]:
    module_path, class_name = identifier.split("@", 1)
    module_candidates = [f"models.{module_path}", module_path]
    with temporary_sys_paths(repo_root, repo_root / "models"):
        last_error: Exception | None = None
        for module_name in module_candidates:
            try:
                module = importlib.import_module(module_name)
                return getattr(module, class_name)
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error


def move_nested_tensors_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if is_dataclass(value) and not isinstance(value, type):
        moved = {item.name: move_nested_tensors_to_device(getattr(value, item.name), device) for item in fields(value)}
        return type(value)(**moved)
    if isinstance(value, dict):
        return {key: move_nested_tensors_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_nested_tensors_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_nested_tensors_to_device(item, device) for item in value)
    return value


def dependency_status() -> dict[str, str]:
    modules = ("einops", "xformers", "stablemax")
    status: dict[str, str] = {}
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            status[module_name] = str(getattr(module, "__version__", "installed"))
        except Exception as exc:
            status[module_name] = f"missing_or_not_needed: {type(exc).__name__}"
    return status


def default_repo_root() -> Path:
    if DEFAULT_KAGGLE_REPO.exists():
        return DEFAULT_KAGGLE_REPO
    return DEFAULT_LOCAL_REPO


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated Samsung TRM checkpoint load and forward test.")
    parser.add_argument("--repo-root", default=str(default_repo_root()), help="TinyRecursiveModels repo root.")
    parser.add_argument("--checkpoint", default=None, help="TRM checkpoint path. Defaults to repo/checkpoints/TRM public checkpoint.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N.")
    parser.add_argument("--seq-len", type=int, default=256, help="Dummy input sequence length.")
    parser.add_argument("--forward-dtype", default="float16", help="Model forward dtype for isolated test.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else repo_root / "checkpoints" / DEFAULT_CHECKPOINT_NAME
    requested_device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Repo root: {repo_root}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Requested device resolved to: {requested_device}")
    print(f"Dependency status: {json.dumps(dependency_status(), sort_keys=True)}")

    sidecar_payload = read_yaml(checkpoint.parent / "all_config.yaml")
    repo_arch_payload = read_yaml(repo_root / "config" / "arch" / "trm.yaml")
    identifier = arch_identifier(sidecar_payload) or arch_identifier(repo_arch_payload) or "recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1"
    print(f"Arch identifier: {identifier}")

    with temporary_sys_paths(repo_root, repo_root / "models"):
        try:
            from recursive_reasoning.trm import TinyRecursiveModel  # type: ignore

            try:
                default_model = TinyRecursiveModel()  # type: ignore[operator]
                print(f"TinyRecursiveModel default init OK: {type(default_model)}")
            except Exception as exc:
                print(f"TinyRecursiveModel exists but default init failed: {type(exc).__name__}: {exc}")
        except Exception as exc:
            print(f"TinyRecursiveModel default path unavailable as expected for this repo: {type(exc).__name__}: {exc}")

    payload = load_checkpoint(checkpoint)
    raw_state = extract_state_dict(payload)
    normalized_state = normalize_state_dict(raw_state)
    print(f"Checkpoint tensor count: {len(raw_state)}")
    print(f"First raw key: {next(iter(raw_state))}")
    print(f"First normalized key: {next(iter(normalized_state))}")

    config = build_config(repo_root, checkpoint, normalized_state, seq_len=args.seq_len, forward_dtype=args.forward_dtype)
    print(f"Config: {json.dumps(json_safe(config), sort_keys=True)}")

    model_cls = import_model_class(repo_root, identifier)
    model = model_cls(config)
    raw_incompatible = model.load_state_dict(raw_state, strict=False)
    print(
        "Raw load strict=False result: "
        f"missing={len(raw_incompatible.missing_keys)} unexpected={len(raw_incompatible.unexpected_keys)}"
    )

    normalized_incompatible = model.load_state_dict(normalized_state, strict=False)
    print(
        "Normalized load strict=False result: "
        f"missing={len(normalized_incompatible.missing_keys)} unexpected={len(normalized_incompatible.unexpected_keys)}"
    )

    model = model.to(requested_device)
    model.eval()
    model_device = next(model.parameters()).device
    print(f"Model class: {model.__class__.__name__}")
    print(f"Model device: {model_device}")
    print(f"Model expects input type: {model.forward.__code__.co_varnames[:5]}")

    vocab_size = max(int(config["vocab_size"]), 1)
    puzzle_count = max(int(config["num_puzzle_identifiers"]), 1)
    dummy = {
        "inputs": torch.randint(0, vocab_size, (1, int(config["seq_len"])), dtype=torch.int64, device=requested_device),
        "labels": torch.randint(0, vocab_size, (1, int(config["seq_len"])), dtype=torch.int64, device=requested_device),
        "puzzle_identifiers": torch.zeros(1, dtype=torch.int64, device=requested_device).clamp(max=puzzle_count - 1),
    }
    print(f"Dummy input device: {dummy['inputs'].device}")

    try:
        with torch.no_grad():
            model(**dummy)  # type: ignore[misc]
    except Exception as exc:
        print(f"Forward **dummy is unsupported for this model: {type(exc).__name__}: {exc}")

    with torch.no_grad():
        with torch.device(requested_device):
            carry = model.initial_carry(dummy)
        carry = move_nested_tensors_to_device(carry, requested_device)
        carry, output = model(carry, dummy)

    logits = output["logits"]
    print(f"Forward OK: output_type={type(output).__name__} logits_shape={tuple(logits.shape)} logits_dtype={logits.dtype}")
    print("ISOLATED_RESULT: " + json.dumps({"status": "PASS", "model_class": identifier, "logits_shape": list(logits.shape)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        traceback.print_exc()
        print("ISOLATED_RESULT: " + json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        raise SystemExit(1)
