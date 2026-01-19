from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol

import torch


@dataclass(frozen=True)
class Batch:
    x: torch.Tensor
    txt_ids: Optional[torch.Tensor]
    txt_mask: Optional[torch.Tensor]
    domain: str
    md5: Optional[list[str]] = None


@dataclass(frozen=True)
class PreparedBatch:
    x: torch.Tensor
    txt_ids: Optional[torch.Tensor]
    txt_mask: Optional[torch.Tensor]


class Domain(Protocol):
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
