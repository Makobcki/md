from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import os
import random

import numpy as np

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name].data)


def save_ckpt(path: str, obj: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def load_ckpt(path: str, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device)


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
