from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .norms import RMSNorm
from .pos_embed import apply_2d_rope


class JointAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, attn_dropout: float = 0.0, qk_norm: bool = True) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.attn_dropout = float(attn_dropout)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        return x.reshape(b, n, self.num_heads, d // self.num_heads).transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rope_grid_hw: Optional[tuple[int, int]] = None,
        rope_start: int = 0,
        rope_length: int = 0,
    ) -> torch.Tensor:
        qh = self.q_norm(self._shape(q))
        kh = self.k_norm(self._shape(k))
        if rope_grid_hw is not None and rope_length > 0:
            qh, kh = apply_2d_rope(
                qh,
                kh,
                grid_hw=rope_grid_hw,
                start=rope_start,
                length=rope_length,
            )
        vh = self._shape(v)
        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, :].to(dtype=torch.bool, device=q.device)
        out = F.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        return out.transpose(1, 2).reshape(q.shape[0], q.shape[1], self.hidden_dim)

