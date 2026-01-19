from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

def unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "_orig_mod"):
        return getattr(model, "_orig_mod")
    if hasattr(model, "module"):
        return getattr(model, "module")
    return model

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
                self.shadow[name] = p.detach().clone()
                continue

            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        base = unwrap_model(model)
        for name, p in base.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name].data)
