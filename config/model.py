from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ModelConfig:
    base_channels: int
    channel_mults: Tuple[int, ...]
    num_res_blocks: int
    dropout: float
    attn_resolutions: Tuple[int, ...]
    attn_heads: int
    attn_head_dim: int
    text_dim: int
    text_layers: int
    text_heads: int
    text_max_len: int
    use_scale_shift_norm: bool
    grad_checkpointing: bool
    use_text_conditioning: bool
    self_conditioning: bool
