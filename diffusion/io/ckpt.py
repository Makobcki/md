from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable

import os

import torch
import torch.nn as nn


_KNOWN_PREFIXES = ("_orig_mod.", "module.")
CKPT_FORMAT_VERSION = 2


def _torch_load(path: str | Path, map_location: torch.device | str):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Older PyTorch has no weights_only argument. These project checkpoints
        # are trusted local training artifacts, not arbitrary uploads.
        return torch.load(path, map_location=map_location)
    except (RuntimeError, pickle.UnpicklingError):
        if os.environ.get("MD_ALLOW_UNSAFE_TORCH_LOAD", "1") not in {"1", "true", "True", "yes"}:
            raise
        # Backward compatibility for legacy project checkpoints containing Python
        # objects unsupported by weights_only.
        return torch.load(path, map_location=map_location, weights_only=False)


def _checkpoint_metadata_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".metadata.json")


def _write_checkpoint_metadata_sidecar(path: Path, obj: dict) -> None:
    metadata = obj.get("metadata") if isinstance(obj, dict) else None
    if not isinstance(metadata, dict):
        return
    payload = {
        "metadata": metadata,
        "cfg": obj.get("cfg", {}),
        "step": obj.get("step", metadata.get("step")),
        "architecture": obj.get("architecture", metadata.get("architecture")),
        "objective": obj.get("objective", metadata.get("objective")),
        "prediction_type": obj.get("prediction_type", metadata.get("prediction_type")),
    }
    sidecar = _checkpoint_metadata_sidecar(path)
    tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    _fsync_file(tmp)
    os.replace(tmp, sidecar)
    _fsync_dir(sidecar.parent)


def _strip_prefixes(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in _KNOWN_PREFIXES:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key


def sanitize_state_dict(sd: Any) -> Any:
    if not isinstance(sd, dict):
        return sd

    out: Dict[str, Any] = {}
    for k, v in sd.items():
        nk = _strip_prefixes(str(k))
        if nk not in out or str(k) == nk:
            out[nk] = v
    return out


def sanitize_ckpt(ck: Any) -> Any:
    if not isinstance(ck, dict):
        return ck

    if "model" in ck:
        ck["model"] = sanitize_state_dict(ck["model"])
    if "ema" in ck and isinstance(ck["ema"], dict):
        ck["ema"] = sanitize_state_dict(ck["ema"])
    return ck


def _build_state_dict_mapping(sd: dict) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in sd.items():
        nk = _strip_prefixes(str(k))
        if nk not in out or str(k) == nk:
            out[nk] = v
    return out


def normalize_state_dict_for_keys(state_dict: dict, target_keys: Iterable[str], name: str) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict
    target_keys_list = [str(k) for k in target_keys]
    target_stripped = {_strip_prefixes(k) for k in target_keys_list}
    source_map = _build_state_dict_mapping(state_dict)

    missing = [k for k in target_keys_list if _strip_prefixes(k) not in source_map]
    unexpected = [k for k in source_map if k not in target_stripped]

    if missing or unexpected:
        missing_examples = ", ".join(sorted({_strip_prefixes(k) for k in missing})[:10])
        unexpected_examples = ", ".join(sorted(unexpected)[:10])
        raise RuntimeError(
            f"{name} state_dict mismatch after prefix normalization: "
            f"missing={len(missing)}, unexpected={len(unexpected)}. "
            f"Missing examples: {missing_examples or 'none'}. "
            f"Unexpected examples: {unexpected_examples or 'none'}."
        )

    return {k: source_map[_strip_prefixes(k)] for k in target_keys_list}


def normalize_state_dict_for_model(state_dict: dict, model: nn.Module, name: str = "model") -> dict:
    return normalize_state_dict_for_keys(state_dict, model.state_dict().keys(), name)


def load_ckpt(path: str, device: torch.device) -> dict:
    ck = _torch_load(path, map_location=device)
    ck = sanitize_ckpt(ck)
    if isinstance(ck, dict):
        format_version = int(ck.get("format_version", 1))
        if format_version > CKPT_FORMAT_VERSION:
            raise RuntimeError(f"Unsupported checkpoint format_version={format_version}.")
        if "format_version" not in ck:
            ck["format_version"] = 1
        if format_version == 1:
            ck = _upgrade_ckpt_v1_to_v2(ck)
        ck = _normalize_ckpt_fields(ck)
    return ck


def save_ckpt(path: str, obj: dict) -> None:
    obj = sanitize_ckpt(obj)
    if isinstance(obj, dict):
        if "format_version" not in obj:
            obj["format_version"] = CKPT_FORMAT_VERSION
        obj = _normalize_ckpt_fields(obj)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(obj, tmp)
    _fsync_file(tmp)
    os.replace(tmp, p)
    _fsync_dir(p.parent)
    _write_checkpoint_metadata_sidecar(p, obj)


def resolve_resume_path(resume: str, out_dir: Path) -> str:
    resume = str(resume).strip()
    if not resume:
        return ""
    checkpoint_dir = out_dir / "checkpoints"
    if resume == "latest":
        candidates = (checkpoint_dir / "latest.pt", out_dir / "latest.pt", out_dir / "ckpt_latest.pt")
        for path in candidates:
            if path.exists():
                return str(path)
        ckpts = sorted(checkpoint_dir.glob("step_*.pt")) or sorted(out_dir.glob("step_*.pt")) or sorted(out_dir.glob("ckpt_*.pt"))
        ckpts = [p for p in ckpts if p.name not in {"ckpt_latest.pt", "ckpt_final.pt"}]
        if ckpts:
            return str(ckpts[-1])
        raise RuntimeError("No checkpoints found for resume=latest.")
    if resume.isdigit():
        step = int(resume)
        candidates = (checkpoint_dir / f"step_{step:06d}.pt", out_dir / f"step_{step:06d}.pt", out_dir / f"ckpt_{step:07d}.pt")
        for path in candidates:
            if path.exists():
                return str(path)
        raise RuntimeError(f"Checkpoint not found for step {step}: {candidates[0]}")
    return resume


def strip_state_dict_prefixes(sd: dict, prefixes: Iterable[str] = ("_orig_mod.", "module.")) -> dict:
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def _upgrade_ckpt_v1_to_v2(ck: dict) -> dict:
    ck = dict(ck)
    ck["format_version"] = CKPT_FORMAT_VERSION
    return ck


def _normalize_ckpt_fields(ck: dict) -> dict:
    ck = dict(ck)
    if "opt" not in ck and "optimizer" in ck:
        ck["opt"] = ck["optimizer"]
    if "meta" not in ck:
        ck["meta"] = {}
    return ck


def _fsync_file(path: Path) -> None:
    try:
        with path.open("rb") as f:
            os.fsync(f.fileno())
    except Exception:
        return


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except Exception:
        return
    try:
        os.fsync(fd)
    except Exception:
        pass
    finally:
        os.close(fd)
