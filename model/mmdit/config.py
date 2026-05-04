from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MMDiTConfig:
    latent_channels: int = 4
    patch_size: int = 2
    hidden_dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    qk_norm: bool = True
    rms_norm: bool = True
    swiglu: bool = True
    adaln_zero: bool = True
    pos_embed: str = "rope_2d"
    rope_scaling: str = "none"
    rope_base_grid_hw: tuple[int, int] = (32, 32)
    rope_theta: float = 10000.0
    double_stream_blocks: int = 16
    single_stream_blocks: int = 8
    dropout: float = 0.0
    attn_dropout: float = 0.0
    gradient_checkpointing: bool = True
    text_dim: int = 1024
    pooled_dim: int = 1024
    zero_init_final: bool = True

    text_resampler_enabled: bool = False
    text_resampler_num_tokens: int = 128
    text_resampler_depth: int = 2
    text_resampler_mlp_ratio: float = 4.0

    attention_schedule: str = "full"
    early_joint_blocks: int = 0
    late_joint_blocks: int = 0

    source_patch_size: int = 2
    mask_patch_size: int = 2
    control_patch_size: int = 2
    mask_as_source_channel: bool = False
    conditioning_rope: bool = True
    strength_embed: bool = False
    control_type_embed: bool = False
    control_adapter: bool = False
    control_adapter_ratio: float = 0.25
    hierarchical_tokens_enabled: bool = False
    coarse_patch_size: int = 4
    x0_aux_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.latent_channels <= 0:
            raise ValueError("latent_channels must be positive.")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.num_heads <= 0 or self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if self.depth <= 0:
            raise ValueError("depth must be positive.")
        if self.double_stream_blocks < 0 or self.single_stream_blocks < 0:
            raise ValueError("block counts must be non-negative.")
        if self.double_stream_blocks + self.single_stream_blocks != self.depth:
            raise ValueError("double_stream_blocks + single_stream_blocks must equal depth.")
        if self.pos_embed not in {"rope_2d", "sincos_2d", "none"}:
            raise ValueError("pos_embed must be one of: rope_2d, sincos_2d, none.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MMDiTConfig":
        model = data.get("model", data)
        text = data.get("text", {})
        kwargs = {
            "latent_channels": int(data.get("latent_channels", model.get("latent_channels", 4))),
            "patch_size": int(data.get("latent_patch_size", model.get("patch_size", 2))),
            "hidden_dim": int(model.get("hidden_dim", 1024)),
            "depth": int(model.get("depth", 24)),
            "num_heads": int(model.get("num_heads", 16)),
            "mlp_ratio": float(model.get("mlp_ratio", 4.0)),
            "qk_norm": bool(model.get("qk_norm", True)),
            "rms_norm": bool(model.get("rms_norm", True)),
            "swiglu": bool(model.get("swiglu", True)),
            "adaln_zero": bool(model.get("adaln_zero", True)),
            "pos_embed": str(model.get("pos_embed", "rope_2d")),
            "rope_scaling": str(rope_cfg.get("scaling", model.get("rope_scaling", "none"))),
            "rope_base_grid_hw": base_grid,
            "rope_theta": float(rope_cfg.get("theta", model.get("rope_theta", 10000.0))),
            "double_stream_blocks": int(model.get("double_stream_blocks", 16)),
            "single_stream_blocks": int(model.get("single_stream_blocks", 8)),
            "dropout": float(model.get("dropout", 0.0)),
            "attn_dropout": float(model.get("attn_dropout", 0.0)),
            "gradient_checkpointing": bool(model.get("gradient_checkpointing", True)),
            "text_dim": int(text.get("text_dim", data.get("text_dim", 1024))),
            "pooled_dim": int(text.get("pooled_dim", data.get("pooled_dim", 1024))),
            "zero_init_final": bool(model.get("zero_init_final", True)),
            "text_resampler_enabled": bool(text.get("resampler", {}).get("enabled", data.get("text_resampler_enabled", model.get("text_resampler_enabled", False)))) if isinstance(text.get("resampler", {}), dict) else bool(data.get("text_resampler_enabled", model.get("text_resampler_enabled", False))),
            "text_resampler_num_tokens": int(text.get("resampler", {}).get("num_tokens", data.get("text_resampler_num_tokens", model.get("text_resampler_num_tokens", 128)))) if isinstance(text.get("resampler", {}), dict) else int(data.get("text_resampler_num_tokens", model.get("text_resampler_num_tokens", 128))),
            "text_resampler_depth": int(text.get("resampler", {}).get("depth", data.get("text_resampler_depth", model.get("text_resampler_depth", 2)))) if isinstance(text.get("resampler", {}), dict) else int(data.get("text_resampler_depth", model.get("text_resampler_depth", 2))),
            "text_resampler_mlp_ratio": float(text.get("resampler", {}).get("mlp_ratio", data.get("text_resampler_mlp_ratio", model.get("text_resampler_mlp_ratio", 4.0)))) if isinstance(text.get("resampler", {}), dict) else float(data.get("text_resampler_mlp_ratio", model.get("text_resampler_mlp_ratio", 4.0))),
            "attention_schedule": str(model.get("attention_schedule", "full")),
            "early_joint_blocks": int(model.get("early_joint_blocks", 0)),
            "late_joint_blocks": int(model.get("late_joint_blocks", 0)),
            "source_patch_size": int(model.get("source_patch_size", model.get("conditioning_tokens", {}).get("source_patch_size", model.get("patch_size", 2))) if isinstance(model.get("conditioning_tokens", {}), dict) else model.get("source_patch_size", model.get("patch_size", 2))),
            "mask_patch_size": int(model.get("mask_patch_size", model.get("conditioning_tokens", {}).get("mask_patch_size", model.get("patch_size", 2))) if isinstance(model.get("conditioning_tokens", {}), dict) else model.get("mask_patch_size", model.get("patch_size", 2))),
            "control_patch_size": int(model.get("control_patch_size", model.get("conditioning_tokens", {}).get("control_patch_size", model.get("patch_size", 2))) if isinstance(model.get("conditioning_tokens", {}), dict) else model.get("control_patch_size", model.get("patch_size", 2))),
            "mask_as_source_channel": bool(model.get("mask_as_source_channel", model.get("conditioning_tokens", {}).get("mask_as_source_channel", False))) if isinstance(model.get("conditioning_tokens", {}), dict) else bool(model.get("mask_as_source_channel", False)),
            "conditioning_rope": bool(model.get("conditioning_rope", True)),
            "strength_embed": bool(model.get("strength_embed", data.get("strength_embed", False))),
            "control_type_embed": bool(model.get("control_type_embed", model.get("control", {}).get("type_embed", False))) if isinstance(model.get("control", {}), dict) else bool(model.get("control_type_embed", False)),
            "control_adapter": bool(model.get("control_adapter", model.get("control", {}).get("adapter", False))) if isinstance(model.get("control", {}), dict) else bool(model.get("control_adapter", False)),
            "control_adapter_ratio": float(model.get("control_adapter_ratio", model.get("control", {}).get("adapter_ratio", 0.25))) if isinstance(model.get("control", {}), dict) else float(model.get("control_adapter_ratio", 0.25)),
            "hierarchical_tokens_enabled": bool(model.get("hierarchical_tokens_enabled", model.get("hierarchical", {}).get("enabled", False))) if isinstance(model.get("hierarchical", {}), dict) else bool(model.get("hierarchical_tokens_enabled", False)),
            "coarse_patch_size": int(model.get("coarse_patch_size", model.get("hierarchical", {}).get("coarse_patch_size", 4))) if isinstance(model.get("hierarchical", {}), dict) else int(model.get("coarse_patch_size", 4)),
            "x0_aux_weight": float(data.get("x0_aux_weight", data.get("loss", {}).get("x0_aux_weight", 0.0))) if isinstance(data.get("loss", {}), dict) else float(data.get("x0_aux_weight", 0.0)),
        }
        return cls(**kwargs)
