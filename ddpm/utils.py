from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import os
import random

import numpy as np

import torch
import torch.nn as nn
from typing import Any, Dict, Mapping, MutableMapping


_KNOWN_PREFIXES = ("_orig_mod.", "module.")

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
    return ck


def save_ckpt(path: str, obj: dict) -> None:
    # нормализуем перед сохранением → новые ckpt будут “чистыми”
    obj = sanitize_ckpt(obj)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, p)



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


def build_run_metadata() -> Dict[str, Any]:
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    }

def strip_state_dict_prefixes(sd: dict, prefixes=("_orig_mod.", "module.")) -> dict:
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out
