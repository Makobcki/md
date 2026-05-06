from __future__ import annotations

from typing import Any, Callable

from .mmdit import MMDiTConfig, MMDiTFlowModel


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
    return str(model.get("family", "mmdit") or "mmdit")


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


MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "mmdit": build_mmdit,
    "flux_like": _build_flux_like,
}


def build_model(config: Any) -> object:
    """Build a model using ``model.family`` semantics."""

    family = model_family(config)
    try:
        builder = MODEL_REGISTRY[family]
    except KeyError as exc:
        allowed = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model family {family!r}. Allowed: {allowed}.") from exc
    return builder(config)
