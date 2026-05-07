from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .contracts import (
    ALLOWED_MODEL_FAMILIES,
    ModelCapabilities,
    ModelRuntimeContract,
    ModelSpec,
    build_model_capabilities,
    build_model_spec,
    build_runtime_contract,
)
from .mmdit import MMDiTConfig, MMDiTFlowModel
from .pixart_sigma import PixArtSigmaConfig, PixArtSigmaRFModel
from .var import VARConfig, VARTransformer

ModelBuilder = Callable[[Any], object]


def _as_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, dict):
        return dict(config)
    if hasattr(config, "to_dict"):
        value = config.to_dict()
        if isinstance(value, dict):
            return value
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    raise TypeError(f"Unsupported model config type: {type(config).__name__}")


def _model_section(config: Any) -> dict[str, Any]:
    data = _as_dict(config)
    model = data.get("model", {})
    return model if isinstance(model, dict) else {}


def model_family(config: Any) -> str:
    """Return semantic model family from a nested config."""
    model = _model_section(config)
    if "family" in model:
        return str(model.get("family") or "mmdit")
    if hasattr(config, "model_family"):
        return str(getattr(config, "model_family") or "mmdit")
    return "mmdit"


def build_mmdit(config: Any) -> MMDiTFlowModel:
    """Build an MMDiT rectified-flow model from dict/TrainConfig/MMDiTConfig."""
    if isinstance(config, MMDiTConfig):
        mmdit_config = config
    else:
        mmdit_config = MMDiTConfig.from_dict(_as_dict(config))
    return MMDiTFlowModel(mmdit_config)


def _build_flux_like(_config: Any) -> object:
    raise NotImplementedError(
        "model.family='flux_like' is registered as a configuration extension point, "
        "but no Flux-like builder is implemented yet."
    )


def _build_pixart_sigma(config: Any) -> PixArtSigmaRFModel:
    data = _as_dict(config)
    model = data.get("model", {}) if isinstance(data.get("model", {}), dict) else {}
    architecture = (
        model.get("architecture", {}) if isinstance(model.get("architecture", {}), dict) else {}
    )
    cfg = PixArtSigmaConfig(
        latent_channels=int(data.get("latent_channels", architecture.get("latent_channels", 4))),
        patch_size=int(data.get("latent_patch_size", architecture.get("patch_size", 2))),
        hidden_size=int(data.get("hidden_dim", architecture.get("hidden_size", 1152))),
        depth=int(data.get("depth", architecture.get("depth", 28))),
        num_heads=int(data.get("num_heads", architecture.get("num_heads", 16))),
        mlp_ratio=float(data.get("mlp_ratio", architecture.get("mlp_ratio", 4.0))),
        qk_norm=bool(data.get("qk_norm", architecture.get("qk_norm", True))),
        caption_channels=int(
            data.get("caption_channels", architecture.get("caption_channels", 4096))
        ),
        cross_attention_dim=int(
            data.get("cross_attention_dim", architecture.get("cross_attention_dim", 1152))
        ),
        max_text_tokens=int(data.get("max_text_tokens", architecture.get("max_text_tokens", 300))),
    )
    return PixArtSigmaRFModel(cfg)


def _build_var(config: Any) -> VARTransformer:
    data = _as_dict(config)
    model = data.get("model", {}) if isinstance(data.get("model", {}), dict) else {}
    architecture = (
        model.get("architecture", {}) if isinstance(model.get("architecture", {}), dict) else {}
    )
    tokenizer = model.get("tokenizer", {}) if isinstance(model.get("tokenizer", {}), dict) else {}
    schedule = data.get("scale_schedule", architecture.get("scale_schedule", (1, 2, 3, 4)))
    cfg = VARConfig(
        codebook_size=int(data.get("codebook_size", tokenizer.get("codebook_size", 4096))),
        hidden_size=int(data.get("hidden_dim", architecture.get("hidden_size", 1024))),
        depth=int(data.get("depth", architecture.get("depth", 16))),
        num_heads=int(data.get("num_heads", architecture.get("num_heads", 16))),
        mlp_ratio=float(data.get("mlp_ratio", architecture.get("mlp_ratio", 4.0))),
        scale_schedule=tuple(int(v) for v in schedule),
        max_token_length=int(
            data.get("max_token_length", architecture.get("max_token_length", 680))
        ),
    )
    return VARTransformer(cfg)


MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "mmdit": build_mmdit,
    "flux_like": _build_flux_like,
    "pixart_sigma": _build_pixart_sigma,
    "var": _build_var,
}


def build_model(config: Any) -> object:
    """Build a model using ``model.family`` semantics."""
    family = model_family(config)
    try:
        builder = MODEL_REGISTRY[family]
    except KeyError as exc:
        allowed = ", ".join(ALLOWED_MODEL_FAMILIES)
        raise ValueError(f"Unknown model family {family!r}. Allowed: {allowed}.") from exc
    return builder(config)


def get_allowed_families() -> tuple[str, ...]:
    return ALLOWED_MODEL_FAMILIES


__all__ = [
    "MODEL_REGISTRY",
    "ModelCapabilities",
    "ModelRuntimeContract",
    "ModelSpec",
    "build_model",
    "build_model_capabilities",
    "build_model_spec",
    "build_runtime_contract",
    "get_allowed_families",
    "model_family",
]
