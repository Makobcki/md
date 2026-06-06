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

    def __post_init__(self) -> None:
        schedule = tuple(int(scale) for scale in self.scale_schedule)
        object.__setattr__(self, "scale_schedule", schedule)
        if int(self.codebook_size) <= 0:
            raise ValueError("VARConfig.codebook_size must be positive.")
        if int(self.hidden_size) <= 0:
            raise ValueError("VARConfig.hidden_size must be positive.")
        if int(self.depth) <= 0:
            raise ValueError("VARConfig.depth must be positive.")
        if int(self.num_heads) <= 0 or int(self.hidden_size) % int(self.num_heads) != 0:
            raise ValueError("VARConfig.hidden_size must be divisible by num_heads.")
        if float(self.mlp_ratio) <= 0.0:
            raise ValueError("VARConfig.mlp_ratio must be positive.")
        if not schedule or any(scale <= 0 for scale in schedule):
            raise ValueError("VARConfig.scale_schedule must contain positive integers.")
        token_count = sum(scale * scale for scale in schedule)
        if int(self.max_token_length) < token_count:
            raise ValueError("VARConfig.max_token_length must cover scale_schedule token count.")


def _scale_token_count(scale: int) -> int:
    return int(scale) * int(scale)


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

    def _validate_tokens(self, tokens: list[torch.Tensor]) -> None:
        if not tokens:
            raise ValueError("VARTransformer requires at least one token scale.")
        if len(tokens) > len(self.cfg.scale_schedule):
            raise ValueError("VAR token scale count exceeds configured scale_schedule.")
        batch_size = int(tokens[0].shape[0]) if tokens[0].ndim >= 1 else -1
        for scale_idx, scale_tokens in enumerate(tokens):
            if scale_tokens.ndim != 2:
                raise ValueError("VAR tokens must have shape [batch, tokens_per_scale].")
            if int(scale_tokens.shape[0]) != batch_size:
                raise ValueError("VAR token scales must share the same batch size.")
            if scale_tokens.dtype != torch.long:
                raise ValueError("VAR tokens must use dtype torch.long.")
            expected_len = _scale_token_count(self.cfg.scale_schedule[scale_idx])
            if int(scale_tokens.shape[1]) != expected_len:
                got_len = int(scale_tokens.shape[1])
                raise ValueError(
                    "VAR token scale length does not match scale_schedule: "
                    f"scale_idx={scale_idx} expected={expected_len} got={got_len}."
                )
            if scale_tokens.numel() == 0:
                continue
            min_token = int(scale_tokens.min().item())
            max_token = int(scale_tokens.max().item())
            if min_token < 0 or max_token >= int(self.cfg.codebook_size):
                raise ValueError("VAR token ids must be in [0, codebook_size).")

    def forward(self, tokens: list[torch.Tensor], *, predict_next: bool = False) -> torch.Tensor:
        self._validate_tokens(tokens)
        device = tokens[0].device
        pieces: list[torch.Tensor] = []
        prefix_tokens = tokens if predict_next or len(tokens) == 1 else tokens[:-1]
        for scale_idx, scale_tokens in enumerate(prefix_tokens):
            emb = self.token_embedding(scale_tokens)
            emb = emb + self.scale_embedding.weight[scale_idx].view(1, 1, -1)
            pieces.append(emb)
        prefix = torch.cat(pieces, dim=1)
        if predict_next:
            target_idx = len(tokens)
            if target_idx >= len(self.cfg.scale_schedule):
                raise ValueError("VAR predict_next requested after the final configured scale.")
            target_len = _scale_token_count(self.cfg.scale_schedule[target_idx])
        elif len(tokens) > 1:
            target_len = int(tokens[-1].shape[1])
        else:
            next_idx = min(len(tokens), len(self.cfg.scale_schedule) - 1)
            target_len = _scale_token_count(self.cfg.scale_schedule[next_idx])
        if target_len > int(self.cfg.max_token_length):
            raise ValueError("VAR target length exceeds max_token_length.")
        if prefix.shape[1] < target_len:
            repeat = target_len - prefix.shape[1]
            prefix = torch.cat([prefix, prefix[:, -1:].expand(-1, repeat, -1)], dim=1)
        hidden = prefix[:, -target_len:]
        position_embedding = self.position_embedding[:, :target_len].to(
            device=device,
            dtype=hidden.dtype,
        )
        hidden = hidden + position_embedding
        return self.head(self.norm(self.blocks(hidden)))


def next_scale_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("VAR logits must have shape [batch, tokens_per_scale, codebook_size].")
    if target.ndim != 2:
        raise ValueError("VAR target tokens must have shape [batch, tokens_per_scale].")
    if target.dtype != torch.long:
        raise ValueError("VAR target tokens must use dtype torch.long.")
    if tuple(logits.shape[:2]) != tuple(target.shape):
        raise ValueError(
            "VAR logits and target token shapes must match on batch and token dimensions."
        )
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))


def multiscale_next_scale_cross_entropy(
    model: VARTransformer, tokens: list[torch.Tensor]
) -> torch.Tensor:
    if len(tokens) < 2:
        raise ValueError("VAR multiscale loss requires at least two token scales.")
    losses = [
        next_scale_cross_entropy(model(tokens[: scale_idx + 1]), tokens[scale_idx])
        for scale_idx in range(1, len(tokens))
    ]
    return torch.stack(losses).mean()


@torch.no_grad()
def deterministic_decode(
    model: VARTransformer,
    *,
    batch_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    batch = int(batch_size)
    if batch <= 0:
        raise ValueError("batch_size must be positive.")
    out: list[torch.Tensor] = []
    for idx, scale in enumerate(model.cfg.scale_schedule):
        length = _scale_token_count(scale)
        if idx == 0:
            tokens = torch.zeros(batch, length, dtype=torch.long, device=device)
        else:
            logits = model(out, predict_next=True)
            tokens = logits.argmax(dim=-1)
        if int(tokens.shape[1]) != length:
            actual_len = int(tokens.shape[1])
            raise RuntimeError(
                f"VAR decoder produced {actual_len} tokens for scale {idx}, expected {length}."
            )
        out.append(tokens)
    return out
