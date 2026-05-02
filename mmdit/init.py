from __future__ import annotations

from torch import nn


def zero_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

