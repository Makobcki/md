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
        if self.pos_embed == "sincos_2d" and int(self.hidden_dim) % 4 != 0:
            raise ValueError(
                "sincos_2d requires hidden_dim divisible by 4; set hidden_dim to a multiple of 4 or use pos_embed=rope_2d/none."
            )
        if self.rope_scaling not in {"none", "linear", "ntk"}:
            raise ValueError("rope_scaling must be one of: none, linear, ntk.")
        if (
            len(self.rope_base_grid_hw) != 2
            or self.rope_base_grid_hw[0] <= 0
            or self.rope_base_grid_hw[1] <= 0
        ):
            raise ValueError("rope_base_grid_hw must contain two positive integers.")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive.")
        if self.text_resampler_num_tokens <= 0 or self.text_resampler_depth <= 0:
            raise ValueError("text resampler num_tokens/depth must be positive.")
        if self.attention_schedule not in {"full", "hybrid"}:
            raise ValueError("attention_schedule must be full or hybrid.")
        if self.early_joint_blocks < 0 or self.late_joint_blocks < 0:
            raise ValueError("joint block counts must be non-negative.")
        for name, value in {
            "source_patch_size": self.source_patch_size,
            "mask_patch_size": self.mask_patch_size,
            "control_patch_size": self.control_patch_size,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.control_adapter_ratio <= 0:
            raise ValueError("control_adapter_ratio must be positive.")
        if self.coarse_patch_size <= 0:
            raise ValueError("coarse_patch_size must be positive.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MMDiTConfig:
        model = data.get("model", {}) if isinstance(data.get("model", {}), dict) else {}
        architecture = (
            model.get("architecture", {}) if isinstance(model.get("architecture", {}), dict) else {}
        )
        text = data.get("text", {}) if isinstance(data.get("text", {}), dict) else {}
        rope_cfg = model.get("rope", {}) if isinstance(model.get("rope", {}), dict) else {}
        conditioning = (
            model.get("conditioning_tokens", {})
            if isinstance(model.get("conditioning_tokens", {}), dict)
            else {}
        )
        control = model.get("control", {}) if isinstance(model.get("control", {}), dict) else {}
        hierarchical = (
            model.get("hierarchical", {}) if isinstance(model.get("hierarchical", {}), dict) else {}
        )
        resampler = text.get("resampler", {}) if isinstance(text.get("resampler", {}), dict) else {}

        def value(key: str, default: Any, *, arch_key: str | None = None) -> Any:
            return data.get(key, model.get(key, architecture.get(arch_key or key, default)))

        def text_value(key: str, default: Any) -> Any:
            return data.get(key, text.get(key, default))

        raw_base_grid = rope_cfg.get(
            "base_grid", data.get("rope_base_grid_hw", model.get("rope_base_grid_hw", (32, 32)))
        )
        if isinstance(raw_base_grid, int):
            base_grid = (int(raw_base_grid), int(raw_base_grid))
        else:
            seq = list(raw_base_grid)
            if len(seq) != 2:
                raise ValueError("rope base_grid must contain two values.")
            base_grid = (int(seq[0]), int(seq[1]))
        kwargs = {
            "latent_channels": int(value("latent_channels", 4)),
            "patch_size": int(data.get("latent_patch_size", value("patch_size", 2))),
            "hidden_dim": int(value("hidden_dim", 1024, arch_key="hidden_size")),
            "depth": int(value("depth", 24)),
            "num_heads": int(value("num_heads", 16)),
            "mlp_ratio": float(value("mlp_ratio", 4.0)),
            "qk_norm": bool(value("qk_norm", True)),
            "rms_norm": bool(value("rms_norm", True)),
            "swiglu": bool(value("swiglu", True)),
            "adaln_zero": bool(value("adaln_zero", True)),
            "pos_embed": str(value("pos_embed", "rope_2d")),
            "rope_scaling": str(
                data.get(
                    "rope_scaling",
                    rope_cfg.get("scaling", model.get("rope_scaling", "none")),
                )
            ),
            "rope_base_grid_hw": base_grid,
            "rope_theta": float(
                data.get("rope_theta", rope_cfg.get("theta", model.get("rope_theta", 10000.0)))
            ),
            "double_stream_blocks": int(value("double_stream_blocks", 16)),
            "single_stream_blocks": int(value("single_stream_blocks", 8)),
            "dropout": float(value("dropout", 0.0)),
            "attn_dropout": float(value("attn_dropout", 0.0)),
            "gradient_checkpointing": bool(value("gradient_checkpointing", True)),
            "text_dim": int(text_value("text_dim", 1024)),
            "pooled_dim": int(text_value("pooled_dim", 1024)),
            "zero_init_final": bool(value("zero_init_final", True)),
            "text_resampler_enabled": bool(
                data.get(
                    "text_resampler_enabled",
                    resampler.get("enabled", model.get("text_resampler_enabled", False)),
                )
            ),
            "text_resampler_num_tokens": int(
                data.get(
                    "text_resampler_num_tokens",
                    resampler.get("num_tokens", model.get("text_resampler_num_tokens", 128)),
                )
            ),
            "text_resampler_depth": int(
                data.get(
                    "text_resampler_depth",
                    resampler.get("depth", model.get("text_resampler_depth", 2)),
                )
            ),
            "text_resampler_mlp_ratio": float(
                data.get(
                    "text_resampler_mlp_ratio",
                    resampler.get("mlp_ratio", model.get("text_resampler_mlp_ratio", 4.0)),
                )
            ),
            "attention_schedule": str(value("attention_schedule", "full")),
            "early_joint_blocks": int(value("early_joint_blocks", 0)),
            "late_joint_blocks": int(value("late_joint_blocks", 0)),
            "source_patch_size": int(
                data.get(
                    "source_patch_size",
                    model.get(
                        "source_patch_size",
                        conditioning.get("source_patch_size", value("patch_size", 2)),
                    ),
                )
            ),
            "mask_patch_size": int(
                data.get(
                    "mask_patch_size",
                    model.get(
                        "mask_patch_size",
                        conditioning.get("mask_patch_size", value("patch_size", 2)),
                    ),
                )
            ),
            "control_patch_size": int(
                data.get(
                    "control_patch_size",
                    model.get(
                        "control_patch_size",
                        conditioning.get("control_patch_size", value("patch_size", 2)),
                    ),
                )
            ),
            "mask_as_source_channel": bool(
                data.get(
                    "mask_as_source_channel",
                    model.get(
                        "mask_as_source_channel",
                        conditioning.get("mask_as_source_channel", False),
                    ),
                )
            ),
            "conditioning_rope": bool(
                data.get("conditioning_rope", model.get("conditioning_rope", True))
            ),
            "strength_embed": bool(data.get("strength_embed", model.get("strength_embed", False))),
            "control_type_embed": bool(
                data.get(
                    "control_type_embed",
                    model.get("control_type_embed", control.get("type_embed", False)),
                )
            ),
            "control_adapter": bool(
                data.get(
                    "control_adapter",
                    model.get("control_adapter", control.get("adapter", False)),
                )
            ),
            "control_adapter_ratio": float(
                data.get(
                    "control_adapter_ratio",
                    model.get("control_adapter_ratio", control.get("adapter_ratio", 0.25)),
                )
            ),
            "hierarchical_tokens_enabled": bool(
                data.get(
                    "hierarchical_tokens_enabled",
                    model.get("hierarchical_tokens_enabled", hierarchical.get("enabled", False)),
                )
            ),
            "coarse_patch_size": int(
                data.get(
                    "coarse_patch_size",
                    model.get("coarse_patch_size", hierarchical.get("coarse_patch_size", 4)),
                )
            ),
            "x0_aux_weight": float(
                data.get("x0_aux_weight", data.get("loss", {}).get("x0_aux_weight", 0.0))
            )
            if isinstance(data.get("loss", {}), dict)
            else float(data.get("x0_aux_weight", 0.0)),
        }
        return cls(**kwargs)
