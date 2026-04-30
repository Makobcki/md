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
    cross_attn_resolutions: Tuple[int, ...]
    attn_heads: int
    attn_head_dim: int
    self_attn_type: str
    self_attn_window_size: int
    cross_attn_dim: int
    mid_blocks: int
    text_dim: int
    text_layers: int
    text_heads: int
    text_max_len: int
    text_spatial_conditioning: bool
    use_scale_shift_norm: bool
    grad_checkpointing: bool
    checkpoint_resblocks: bool
    checkpoint_attention: bool
    checkpoint_text_encoder: bool
    zero_init_residual: bool
    use_text_conditioning: bool
    self_conditioning: bool
