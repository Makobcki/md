from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ALLOWED_MODEL_FAMILIES: tuple[str, ...] = ("mmdit", "pixart_sigma", "var")


@dataclass(frozen=True)
class ModelSpec:
    family: str
    variant: str
    architecture: str
    objective: str
    prediction_type: str
    input_kind: str
    output_kind: str
    conditioning_kind: str
    supports_training: bool
    supports_sampling: bool
    supports_checkpoint_loading: bool


@dataclass(frozen=True)
class ModelCapabilities:
    train: bool
    sample: bool
    resume: bool
    text_to_image: bool
    image_to_image: bool
    inpaint: bool
    control: bool
    latent_flow_sampler: bool
    autoregressive_sampler: bool
    external_checkpoint_import: bool


@dataclass(frozen=True)
class ModelRuntimeContract:
    family: str
    input_kind: str
    output_kind: str
    batch_kind: str
    objective: str
    sampler_kind: str | None
    requires_text_conditioning: bool
    requires_vae: bool
    requires_tokenizer: bool


def _section(data: Any, key: str) -> dict[str, Any]:
    if hasattr(data, "to_dict"):
        data = data.to_dict()
    elif hasattr(data, "__dict__") and not isinstance(data, dict):
        data = vars(data)
    value = data.get(key, {}) if isinstance(data, dict) else {}
    return value if isinstance(value, dict) else {}


def normalize_family(config: Any) -> str:
    model = _section(config, "model")
    family = model.get("family")
    if family is None and hasattr(config, "model_family"):
        family = getattr(config, "model_family")
    return str(family or "mmdit")


def variant_for(config: Any, family: str) -> str:
    model = _section(config, "model")
    variant = model.get("variant")
    if variant is None and hasattr(config, "model_variant"):
        variant = getattr(config, "model_variant")
    defaults = {
        "mmdit": "mmdit_rf_base",
        "pixart_sigma": "pixart_sigma_512",
        "var": "var_d16",
    }
    return str(variant or defaults[family])


def architecture_for_family(family: str) -> str:
    return {
        "mmdit": "mmdit_rf",
        "pixart_sigma": "pixart_sigma_rf",
        "var": "var_ar",
    }[family]


def build_model_spec(config: Any) -> ModelSpec:
    family = normalize_family(config)
    if family not in ALLOWED_MODEL_FAMILIES:
        allowed = ", ".join(ALLOWED_MODEL_FAMILIES)
        raise ValueError(f"Unknown model family {family!r}. Allowed: {allowed}.")
    if family == "var":
        return ModelSpec(
            family=family,
            variant=variant_for(config, family),
            architecture="var_ar",
            objective="next_scale_prediction",
            prediction_type="token_logits",
            input_kind="discrete_multiscale_tokens",
            output_kind="token_logits",
            conditioning_kind="none",
            supports_training=True,
            supports_sampling=True,
            supports_checkpoint_loading=True,
        )
    return ModelSpec(
        family=family,
        variant=variant_for(config, family),
        architecture=architecture_for_family(family),
        objective="rectified_flow",
        prediction_type="velocity",
        input_kind="latent",
        output_kind="latent_velocity",
        conditioning_kind="text",
        supports_training=True,
        supports_sampling=True,
        supports_checkpoint_loading=True,
    )


def build_model_capabilities(config: Any) -> ModelCapabilities:
    family = build_model_spec(config).family
    if family == "mmdit":
        return ModelCapabilities(True, True, True, True, True, True, True, True, False, False)
    if family == "pixart_sigma":
        return ModelCapabilities(True, True, True, True, False, False, False, True, False, False)
    return ModelCapabilities(True, True, True, True, False, False, False, False, True, False)


def build_runtime_contract(config: Any) -> ModelRuntimeContract:
    spec = build_model_spec(config)
    if spec.family == "var":
        return ModelRuntimeContract(
            family=spec.family,
            input_kind=spec.input_kind,
            output_kind=spec.output_kind,
            batch_kind="multiscale_tokens",
            objective=spec.objective,
            sampler_kind="var_autoregressive",
            requires_text_conditioning=False,
            requires_vae=False,
            requires_tokenizer=True,
        )
    return ModelRuntimeContract(
        family=spec.family,
        input_kind=spec.input_kind,
        output_kind=spec.output_kind,
        batch_kind="latent_text",
        objective=spec.objective,
        sampler_kind="latent_flow",
        requires_text_conditioning=True,
        requires_vae=True,
        requires_tokenizer=False,
    )
