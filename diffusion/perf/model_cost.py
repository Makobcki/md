from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from model.unet.unet import UNet, UNetConfig


@dataclass(frozen=True)
class AttentionCostProfile:
    self_blocks: int
    cross_blocks: int
    self_estimated_flops: int
    cross_estimated_flops: int
    max_self_score_memory_bytes: int
    max_cross_score_memory_bytes: int


def _count_params(module: torch.nn.Module, *, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def _contains(resolutions: Iterable[int], value: int) -> bool:
    return int(value) in {int(r) for r in resolutions}


def _self_attention_enabled(placement: str, stage: str) -> bool:
    if placement == "all":
        return True
    if placement == "mid_down":
        return stage in {"down", "mid"}
    if placement == "mid_only":
        return stage == "mid"
    return False


def _attention_flops(batch: int, heads: int, query_tokens: int, key_tokens: int, head_dim: int) -> int:
    # QK^T and AV; multiply-add counted as two FLOPs.
    return int(4 * batch * heads * query_tokens * key_tokens * head_dim)


def _score_memory_bytes(batch: int, heads: int, query_tokens: int, key_tokens: int, dtype_bytes: int) -> int:
    return int(batch * heads * query_tokens * key_tokens * dtype_bytes)


def _iter_unet_attention_sites(cfg: UNetConfig, latent_side: int):
    resolutions = [max(int(latent_side) // (2**level), 1) for level in range(len(cfg.channel_mults))]
    for res in resolutions:
        for _ in range(int(cfg.num_res_blocks)):
            yield "down", res
    mid_res = resolutions[-1]
    yield "mid", mid_res
    for _ in range(max(int(cfg.mid_blocks) - 1, 0)):
        yield "mid", mid_res
    for res in reversed(resolutions):
        for _ in range(int(cfg.num_res_blocks)):
            yield "up", res


def build_model_cost_profile(
    *,
    model: UNet,
    cfg: UNetConfig,
    image_size: int,
    mode: str,
    latent_downsample_factor: int,
    batch_size: int,
    text_tokens: int,
    dtype: torch.dtype,
) -> dict:
    side = int(image_size)
    if str(mode) == "latent":
        side = side // int(latent_downsample_factor)
    dtype_bytes = 2 if dtype in {torch.float16, torch.bfloat16} else 4

    self_blocks = 0
    cross_blocks = 0
    self_flops = 0
    cross_flops = 0
    max_self_score_memory = 0
    max_cross_score_memory = 0

    for stage, res in _iter_unet_attention_sites(cfg, side):
        query_tokens = int(res) * int(res)
        if _contains(cfg.attn_resolutions, res) and _self_attention_enabled(cfg.attention_placement, stage):
            self_blocks += 1
            block_flops = _attention_flops(
                int(batch_size),
                int(cfg.attn_heads),
                query_tokens,
                query_tokens,
                int(cfg.attn_head_dim),
            )
            self_flops += block_flops
            max_self_score_memory = max(
                max_self_score_memory,
                _score_memory_bytes(int(batch_size), int(cfg.attn_heads), query_tokens, query_tokens, dtype_bytes),
            )

        cross_resolutions = tuple(cfg.cross_attn_resolutions) or tuple(cfg.attn_resolutions)
        if bool(cfg.use_text_conditioning) and _contains(cross_resolutions, res):
            cross_blocks += 1
            key_tokens = int(text_tokens)
            cross_flops += _attention_flops(
                int(batch_size),
                int(cfg.attn_heads),
                query_tokens,
                key_tokens,
                int(cfg.attn_head_dim),
            )
            max_cross_score_memory = max(
                max_cross_score_memory,
                _score_memory_bytes(int(batch_size), int(cfg.attn_heads), query_tokens, key_tokens, dtype_bytes),
            )

    attention = AttentionCostProfile(
        self_blocks=self_blocks,
        cross_blocks=cross_blocks,
        self_estimated_flops=self_flops,
        cross_estimated_flops=cross_flops,
        max_self_score_memory_bytes=max_self_score_memory,
        max_cross_score_memory_bytes=max_cross_score_memory,
    )
    text_params = _count_params(model.text_enc) if model.text_enc is not None else 0
    return {
        "total_params": int(_count_params(model)),
        "trainable_params": int(_count_params(model, trainable_only=True)),
        "unet_params": int(_count_params(model) - text_params),
        "text_encoder_params": int(text_params),
        "latent_or_image_side": int(side),
        "batch_size": int(batch_size),
        "dtype": str(dtype).replace("torch.", ""),
        "attention": asdict(attention),
    }
