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
    pos_embed: str = "sincos_2d"
    double_stream_blocks: int = 16
    single_stream_blocks: int = 8
    dropout: float = 0.0
    attn_dropout: float = 0.0
    gradient_checkpointing: bool = True
    text_dim: int = 1024
    pooled_dim: int = 1024
    zero_init_final: bool = True

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
            "pos_embed": str(model.get("pos_embed", "sincos_2d")),
            "double_stream_blocks": int(model.get("double_stream_blocks", 16)),
            "single_stream_blocks": int(model.get("single_stream_blocks", 8)),
            "dropout": float(model.get("dropout", 0.0)),
            "attn_dropout": float(model.get("attn_dropout", 0.0)),
            "gradient_checkpointing": bool(model.get("gradient_checkpointing", True)),
            "text_dim": int(text.get("text_dim", data.get("text_dim", 1024))),
            "pooled_dim": int(text.get("pooled_dim", data.get("pooled_dim", 1024))),
            "zero_init_final": bool(model.get("zero_init_final", True)),
        }
        return cls(**kwargs)
