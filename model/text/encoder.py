from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ff import GEGLUFeedForward
from .norms import RMSNorm


class SDPATransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim must be divisible by n_heads (dim={dim}, n_heads={n_heads}).")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout = float(dropout)

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=True)
        self.ff = GEGLUFeedForward(dim, hidden_dim=dim * 4)
        self.drop = nn.Dropout(self.dropout)
        self.attn_gate = nn.Parameter(torch.ones(1))
        self.ff_gate = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        h = self.norm1(x)
        qkv = self.to_qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        key_padding = ~attn_mask
        sdpa_mask = key_padding.unsqueeze(1).unsqueeze(2)
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(b, t, -1)
        x = x + self.drop(self.to_out(attn_out)) * self.attn_gate

        h = self.norm2(x)
        x = x + self.drop(self.ff(h)) * self.ff_gate
        return x


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int, n_layers: int, n_heads: int, max_len: int):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList(
            [SDPATransformerBlock(dim=dim, n_heads=n_heads) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(dim)

    def forward(self, ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, t = ids.shape
        pos = torch.arange(0, t, device=ids.device).unsqueeze(0).expand(b, t)
        x = self.tok(ids) + self.pos(pos)
        for block in self.blocks:
            x = block(x, attn_mask)
        return self.norm(x)
