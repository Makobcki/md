from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class VARConfig:
    codebook_size: int = 4096
    hidden_size: int = 1024
    depth: int = 16
    num_heads: int = 16
    mlp_ratio: float = 4.0
    scale_schedule: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    max_token_length: int = 680


class VARTransformer(nn.Module):
    """Tiny scale-causal VAR transformer for token-pyramid smoke paths."""

    def __init__(self, cfg: VARConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hidden = int(cfg.hidden_size)
        self.token_embedding = nn.Embedding(int(cfg.codebook_size), hidden)
        self.scale_embedding = nn.Embedding(len(cfg.scale_schedule), hidden)
        self.position_embedding = nn.Parameter(torch.zeros(1, int(cfg.max_token_length), hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=int(cfg.num_heads),
            dim_feedforward=max(hidden, int(hidden * float(cfg.mlp_ratio))),
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=int(cfg.depth))
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, int(cfg.codebook_size))

    def forward(self, tokens: list[torch.Tensor]) -> torch.Tensor:
        if not tokens:
            raise ValueError("VARTransformer requires at least one token scale.")
        device = tokens[0].device
        pieces: list[torch.Tensor] = []
        prefix_tokens = tokens[:-1] if len(tokens) > 1 else tokens
        for scale_idx, scale_tokens in enumerate(prefix_tokens):
            emb = self.token_embedding(scale_tokens)
            emb = emb + self.scale_embedding.weight[scale_idx].view(1, 1, -1)
            pieces.append(emb)
        prefix = torch.cat(pieces, dim=1)
        if len(tokens) > 1:
            target_len = int(tokens[-1].shape[1])
        else:
            next_idx = min(len(tokens), len(self.cfg.scale_schedule) - 1)
            target_len = int(self.cfg.scale_schedule[next_idx]) ** 2
        if prefix.shape[1] < target_len:
            repeat = target_len - prefix.shape[1]
            prefix = torch.cat([prefix, prefix[:, -1:].expand(-1, repeat, -1)], dim=1)
        hidden = prefix[:, -target_len:]
        hidden = hidden + self.position_embedding[:, :target_len].to(device=device)
        return self.head(self.norm(self.blocks(hidden)))


def next_scale_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))


@torch.no_grad()
def deterministic_decode(
    model: VARTransformer,
    *,
    batch_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for idx, scale in enumerate(model.cfg.scale_schedule):
        length = int(scale) * int(scale)
        if idx == 0:
            tokens = torch.zeros(batch_size, length, dtype=torch.long, device=device)
        else:
            logits = model(out)
            tokens = logits.argmax(dim=-1)
        out.append(tokens[:, :length])
    return out
