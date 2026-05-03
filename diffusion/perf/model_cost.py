from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelCostProfile:
    """Lightweight static cost profile for the MMDiT rectified-flow model.

    The profile intentionally uses config-level arithmetic only.  It is meant for
    dry-run diagnostics and tests, not for exact profiler-grade FLOP accounting.
    """

    hidden_dim: int
    num_heads: int
    head_dim: int
    depth: int
    double_stream_blocks: int
    single_stream_blocks: int
    attention_sites: int
    double_stream_attention_sites: int
    single_stream_attention_sites: int
    latent_channels: int
    latent_patch_size: int
    latent_height: int
    latent_width: int
    image_tokens: int
    text_tokens: int
    total_tokens: int
    estimated_attention_score_elements: int
    estimated_attention_flops: int
    estimated_mlp_flops: int
    estimated_total_flops: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    def __getitem__(self, key: str) -> int:
        aliases = {
            "num_attention_sites": "attention_sites",
            "attention_blocks": "attention_sites",
            "num_attention_blocks": "attention_sites",
            "total_attention_sites": "attention_sites",
        }
        return self.to_dict()[aliases.get(key, key)]

    @property
    def num_attention_sites(self) -> int:
        return self.attention_sites

    @property
    def attention_blocks(self) -> int:
        return self.attention_sites

    @property
    def num_attention_blocks(self) -> int:
        return self.attention_sites

    @property
    def total_attention_sites(self) -> int:
        return self.attention_sites


def _read_config_value(cfg: Any, *names: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        model = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
        for name in names:
            if name in cfg:
                return cfg[name]
            if name in model:
                return model[name]
        return default
    for name in names:
        if hasattr(cfg, name):
            return getattr(cfg, name)
    return default


def _parse_latent_hw(
    *,
    latent_hw: tuple[int, int] | None,
    image_size: int | tuple[int, int] | None,
    latent_downsample_factor: int,
) -> tuple[int, int]:
    if latent_hw is not None:
        h, w = latent_hw
        return int(h), int(w)
    if image_size is None:
        return 64, 64
    if isinstance(image_size, int):
        h = w = image_size
    else:
        h, w = image_size
    return int(h) // int(latent_downsample_factor), int(w) // int(latent_downsample_factor)


def build_model_cost_profile(
    cfg: Any,
    *,
    latent_hw: tuple[int, int] | None = None,
    image_size: int | tuple[int, int] | None = None,
    text_tokens: int | None = None,
    latent_downsample_factor: int = 8,
) -> ModelCostProfile:
    """Build a deterministic static MMDiT cost profile from a config/model.

    Args:
        cfg: ``MMDiTConfig``, ``TrainConfig``/resolved dict, or an object with
            compatible attributes.  If a model instance is passed, its ``.cfg``
            attribute is used when present.
        latent_hw: Latent spatial size ``(H, W)``.  Defaults to ``64x64``.
        image_size: Pixel image size used to derive latent size when
            ``latent_hw`` is not supplied.
        text_tokens: Number of text tokens.  If omitted, it is inferred from the
            text config when available and otherwise defaults to ``0``.
        latent_downsample_factor: Pixel-to-latent downsample factor used with
            ``image_size``.
    """

    cfg_obj = getattr(cfg, "cfg", cfg)
    hidden_dim = int(_read_config_value(cfg_obj, "hidden_dim", default=1024))
    num_heads = int(_read_config_value(cfg_obj, "num_heads", default=16))
    depth = int(_read_config_value(cfg_obj, "depth", default=24))
    double_blocks = int(_read_config_value(cfg_obj, "double_stream_blocks", default=16))
    single_blocks = int(_read_config_value(cfg_obj, "single_stream_blocks", default=8))
    latent_channels = int(_read_config_value(cfg_obj, "latent_channels", default=4))
    patch_size = int(_read_config_value(cfg_obj, "patch_size", "latent_patch_size", default=2))
    mlp_ratio = float(_read_config_value(cfg_obj, "mlp_ratio", default=4.0))

    if text_tokens is None:
        text_cfg = cfg_obj.get("text", {}) if isinstance(cfg_obj, dict) else getattr(cfg_obj, "text", {})
        if isinstance(text_cfg, dict):
            max_lengths = [int(e.get("max_length", 0)) for e in text_cfg.get("encoders", []) if isinstance(e, dict)]
            text_tokens = sum(max_lengths) if max_lengths else int(text_cfg.get("max_length", 0))
        else:
            text_tokens = int(getattr(text_cfg, "max_length", 0)) if text_cfg is not None else 0
    text_tokens = int(text_tokens)

    latent_h, latent_w = _parse_latent_hw(
        latent_hw=latent_hw,
        image_size=image_size,
        latent_downsample_factor=latent_downsample_factor,
    )
    if patch_size <= 0:
        raise ValueError("latent_patch_size/patch_size must be positive.")
    if latent_h % patch_size != 0 or latent_w % patch_size != 0:
        raise ValueError(
            f"latent size {(latent_h, latent_w)} must be divisible by patch size {patch_size}."
        )
    if num_heads <= 0 or hidden_dim % num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads.")

    image_tokens = (latent_h // patch_size) * (latent_w // patch_size)
    total_tokens = image_tokens + text_tokens
    attention_sites = double_blocks + single_blocks
    head_dim = hidden_dim // num_heads

    # Approximate per-sample attention/MLP cost. This is intentionally coarse but
    # stable and useful for relative comparisons in dry-run output.
    double_score_elems = double_blocks * num_heads * total_tokens * total_tokens
    single_score_elems = single_blocks * num_heads * total_tokens * total_tokens
    attention_score_elements = int(double_score_elems + single_score_elems)
    attention_flops = int(2 * attention_score_elements * head_dim)
    mlp_hidden = int(hidden_dim * mlp_ratio)
    mlp_flops = int(attention_sites * total_tokens * 2 * hidden_dim * mlp_hidden)

    return ModelCostProfile(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        depth=depth,
        double_stream_blocks=double_blocks,
        single_stream_blocks=single_blocks,
        attention_sites=attention_sites,
        double_stream_attention_sites=double_blocks,
        single_stream_attention_sites=single_blocks,
        latent_channels=latent_channels,
        latent_patch_size=patch_size,
        latent_height=latent_h,
        latent_width=latent_w,
        image_tokens=image_tokens,
        text_tokens=text_tokens,
        total_tokens=total_tokens,
        estimated_attention_score_elements=attention_score_elements,
        estimated_attention_flops=attention_flops,
        estimated_mlp_flops=mlp_flops,
        estimated_total_flops=attention_flops + mlp_flops,
    )
