from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol

import torch


@dataclass(frozen=True)
class Batch:
    # Батч доменных данных: x — либо пиксели [B,3,H,W], либо латенты [B,4,h,w].
    x: torch.Tensor
    txt_ids: Optional[torch.Tensor]
    txt_mask: Optional[torch.Tensor]
    domain: str
    md5: Optional[list[str]] = None


@dataclass(frozen=True)
class PreparedBatch:
    # Подготовленный батч: все тензоры на устройстве/в формате домена.
    x: torch.Tensor
    txt_ids: Optional[torch.Tensor]
    txt_mask: Optional[torch.Tensor]


class Domain(Protocol):
    # Контракт домена: пиксели и латенты реализуют одинаковые операции.
    name: str

    def prepare_batch(self, batch: Batch) -> PreparedBatch:
        ...

    def sample_noise_like(self, x0: torch.Tensor) -> torch.Tensor:
        ...

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ...

    def v_target(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ...

    def decode_for_preview(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def shape_info(self, x: torch.Tensor) -> Dict[str, int]:
        ...
