from __future__ import annotations

import torch
from torch import nn

from .attention import JointAttention
from .norms import build_norm
from .blocks import FeedForward


class TextResamplerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float, attn_dropout: float, qk_norm: bool, rms_norm: bool, swiglu: bool) -> None:
        super().__init__()
        self.q_norm = build_norm(hidden_dim, rms_norm=rms_norm)
        self.kv_norm = build_norm(hidden_dim, rms_norm=rms_norm)
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.kv = nn.Linear(hidden_dim, hidden_dim * 2)
        self.attn = JointAttention(hidden_dim, num_heads, attn_dropout, qk_norm)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.ff_norm = build_norm(hidden_dim, rms_norm=rms_norm)
        self.ff = FeedForward(hidden_dim, mlp_ratio, dropout, swiglu)

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        q = self.q(self.q_norm(queries))
        k, v = self.kv(self.kv_norm(tokens)).chunk(2, dim=-1)
        queries = queries + self.out(self.attn(q, k, v, mask=mask))
        queries = queries + self.ff(self.ff_norm(queries))
        return queries


class TextResampler(nn.Module):
    """Perceiver-style text compressor.

    Learned query tokens cross-attend to raw CLIP/T5 tokens and produce a fixed
    number of compact conditioning tokens. The output is dense by construction,
    so downstream joint attention can run without a text padding mask.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_tokens: int = 128,
        depth: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        qk_norm: bool = True,
        rms_norm: bool = True,
        swiglu: bool = True,
    ) -> None:
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.num_tokens = int(num_tokens)
        self.queries = nn.Parameter(torch.randn(1, self.num_tokens, hidden_dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                TextResamplerLayer(hidden_dim, num_heads, mlp_ratio, dropout, attn_dropout, qk_norm, rms_norm, swiglu)
                for _ in range(int(depth))
            ]
        )
        self.norm = build_norm(hidden_dim, rms_norm=rms_norm)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.queries.to(device=tokens.device, dtype=tokens.dtype).expand(tokens.shape[0], -1, -1)
        for layer in self.layers:
            q = layer(q, tokens, mask)
        return self.norm(q)
