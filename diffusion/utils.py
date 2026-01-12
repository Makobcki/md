from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import os
import random

import numpy as np

import torch
import torch.nn as nn


_KNOWN_PREFIXES = ("_orig_mod.", "module.")
CKPT_FORMAT_VERSION = 2

def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Возвращает базовую (не-обёрнутую) модель:
    - torch.compile -> model._orig_mod
    - DDP/DataParallel -> model.module
    """
    if hasattr(model, "_orig_mod"):
        return getattr(model, "_orig_mod")
    if hasattr(model, "module"):
        return getattr(model, "module")
    return model

def _strip_prefixes(key: str) -> str:
    # Убираем префиксы рекурсивно (на случай "_orig_mod.module.")
    changed = True
    while changed:
        changed = False
        for p in _KNOWN_PREFIXES:
            if key.startswith(p):
                key = key[len(p):]
                changed = True
    return key


def sanitize_state_dict(sd: Any) -> Any:
    """
    Нормализует ключи state_dict/EMA:
    - убирает _orig_mod. и module.
    """
    if not isinstance(sd, dict):
        return sd

    out: Dict[str, Any] = {}
    for k, v in sd.items():
        nk = _strip_prefixes(str(k))
        # Если коллизия, предпочитаем "чистый" ключ
        if nk not in out or str(k) == nk:
            out[nk] = v
    return out


def sanitize_ckpt(ck: Any) -> Any:
    """
    Нормализует словарь чекпоинта: model/ema.
    Делает load/save совместимыми между compiled / non-compiled.
    """
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


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}

        base = unwrap_model(model)
        for name, p in base.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        base = unwrap_model(model)
        for name, p in base.named_parameters():
            if not p.requires_grad:
                continue

            if name not in self.shadow:
                # если архитектуру поменяли во время resume — не падаем
                self.shadow[name] = p.detach().clone()
                continue

            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        base = unwrap_model(model)
        for name, p in base.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name].data)



def load_ckpt(path: str, device: torch.device) -> dict:
    ck = torch.load(path, map_location=device)
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
    # нормализуем перед сохранением → новые ckpt будут “чистыми”
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


def resolve_resume_path(resume: str, out_dir: Path) -> str:
    """
    Поддерживает resume форматы: "latest", "step" (число), абсолютный/относительный путь.
    """
    resume = str(resume).strip()
    if not resume:
        return ""
    if resume == "latest":
        latest_path = out_dir / "ckpt_latest.pt"
        if latest_path.exists():
            return str(latest_path)
        ckpts = sorted(out_dir.glob("ckpt_*.pt"))
        if ckpts:
            return str(ckpts[-1])
        raise RuntimeError("No checkpoints found for resume=latest.")
    if resume.isdigit():
        step = int(resume)
        ckpt_path = out_dir / f"ckpt_{step:07d}.pt"
        if not ckpt_path.exists():
            raise RuntimeError(f"Checkpoint not found for step {step}: {ckpt_path}")
        return str(ckpt_path)
    return resume


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


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def build_run_metadata(perf: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
    meta = {
        "torch_version": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": (str(torch.version.cuda) if torch.version.cuda is not None else None),
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    }
    if perf:
        meta["perf"] = dict(perf)
    try:
        import subprocess

        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(Path.cwd()), stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        meta["git_commit"] = None
    return _normalize_metadata(meta)


def _normalize_metadata(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.hex()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _normalize_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_metadata(v) for v in value]
    return str(value)


def strip_state_dict_prefixes(sd: dict, prefixes=("_orig_mod.", "module.")) -> dict:
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out
