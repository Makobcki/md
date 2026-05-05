from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from diffusion.io.ckpt import _torch_load

_REQUIRED_METADATA_FIELDS = (
    "architecture",
    "objective",
    "prediction_type",
    "model_config",
    "text_config",
    "vae_config",
    "flow_config",
    "train_config_hash",
    "dataset_hash",
    "step",
)

_MODEL_COMPAT_FIELDS = (
    "latent_channels",
    "latent_patch_size",
    "hidden_dim",
    "depth",
    "num_heads",
    "mlp_ratio",
    "qk_norm",
    "rms_norm",
    "swiglu",
    "double_stream_blocks",
    "single_stream_blocks",
    "pos_embed",
    "rope_scaling",
    "rope_base_grid_hw",
    "rope_theta",
    "text_resampler_enabled",
    "text_resampler_num_tokens",
    "text_resampler_depth",
    "text_resampler_mlp_ratio",
    "attention_schedule",
    "early_joint_blocks",
    "late_joint_blocks",
    "source_patch_size",
    "mask_patch_size",
    "control_patch_size",
    "mask_as_source_channel",
    "conditioning_rope",
    "strength_embed",
    "control_type_embed",
    "control_adapter",
    "control_adapter_ratio",
    "hierarchical_tokens_enabled",
    "coarse_patch_size",
)

_TOP_LEVEL_COMPAT_FIELDS = (
    "architecture",
    "objective",
)


def stable_config_hash(cfg: dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_text_encoders(items: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for item in items:
        normalized.append(
            {
                "name": str(item.get("name", "")),
                "model_name": str(item.get("model_name", "")),
                "max_length": int(item.get("max_length", 0)),
                "trainable": bool(item.get("trainable", False)),
            }
        )
    return normalized


def _config_text_encoders(cfg: dict) -> list[dict]:
    return list(cfg.get("text_encoders", cfg.get("text", {}).get("encoders", [])) or [])


def _ckpt_metadata(ckpt: dict) -> dict:
    meta = ckpt.get("metadata")
    if isinstance(meta, dict):
        return meta
    # Backward-compatible view for old checkpoints that had fields at top level.
    return {
        "architecture": ckpt.get("architecture", ckpt.get("cfg", {}).get("architecture")),
        "objective": ckpt.get("objective", ckpt.get("cfg", {}).get("objective")),
        "prediction_type": ckpt.get("prediction_type", ckpt.get("cfg", {}).get("prediction_type")),
        "model_config": ckpt.get("model_config", ckpt.get("cfg", {})),
        "text_config": ckpt.get("text_config", ckpt.get("cfg", {}).get("text", {})),
        "vae_config": ckpt.get("vae_config", ckpt.get("vae", {})),
        "flow_config": ckpt.get("flow_config", ckpt.get("cfg", {}).get("flow", {})),
        "train_config_hash": ckpt.get("train_config_hash", ""),
        "dataset_hash": ckpt.get("dataset_hash", ""),
        "step": ckpt.get("step"),
    }


def build_mmdit_checkpoint_metadata(
    *,
    cfg: Any,
    cfg_dict: dict[str, Any],
    step: int,
    text_metadata: dict[str, Any] | None = None,
    dataset_hash: str = "",
) -> dict[str, Any]:
    text_metadata = dict(text_metadata or {})
    model_config = {
        "latent_channels": int(cfg.latent_channels),
        "latent_patch_size": int(cfg.latent_patch_size),
        "hidden_dim": int(cfg.hidden_dim),
        "depth": int(cfg.depth),
        "num_heads": int(cfg.num_heads),
        "mlp_ratio": float(getattr(cfg, "mlp_ratio", 4.0)),
        "qk_norm": bool(getattr(cfg, "qk_norm", True)),
        "rms_norm": bool(getattr(cfg, "rms_norm", True)),
        "swiglu": bool(getattr(cfg, "swiglu", True)),
        "double_stream_blocks": int(cfg.double_stream_blocks),
        "single_stream_blocks": int(cfg.single_stream_blocks),
        "pos_embed": str(cfg.pos_embed),
        "rope_scaling": str(getattr(cfg, "rope_scaling", "none")),
        "rope_base_grid_hw": list(getattr(cfg, "rope_base_grid_hw", (32, 32))),
        "rope_theta": float(getattr(cfg, "rope_theta", 10000.0)),
        "text_dim": int(cfg.text_dim),
        "pooled_dim": int(cfg.pooled_dim),
        "text_resampler_enabled": bool(getattr(cfg, "text_resampler_enabled", False)),
        "text_resampler_num_tokens": int(getattr(cfg, "text_resampler_num_tokens", 128)),
        "text_resampler_depth": int(getattr(cfg, "text_resampler_depth", 2)),
        "text_resampler_mlp_ratio": float(getattr(cfg, "text_resampler_mlp_ratio", 4.0)),
        "attention_schedule": str(getattr(cfg, "attention_schedule", "full")),
        "early_joint_blocks": int(getattr(cfg, "early_joint_blocks", 0)),
        "late_joint_blocks": int(getattr(cfg, "late_joint_blocks", 0)),
        "source_patch_size": int(getattr(cfg, "source_patch_size", cfg.latent_patch_size)),
        "mask_patch_size": int(getattr(cfg, "mask_patch_size", cfg.latent_patch_size)),
        "control_patch_size": int(getattr(cfg, "control_patch_size", cfg.latent_patch_size)),
        "mask_as_source_channel": bool(getattr(cfg, "mask_as_source_channel", False)),
        "conditioning_rope": bool(getattr(cfg, "conditioning_rope", True)),
        "strength_embed": bool(getattr(cfg, "strength_embed", False)),
        "control_type_embed": bool(getattr(cfg, "control_type_embed", False)),
        "control_adapter": bool(getattr(cfg, "control_adapter", False)),
        "control_adapter_ratio": float(getattr(cfg, "control_adapter_ratio", 0.25)),
        "hierarchical_tokens_enabled": bool(getattr(cfg, "hierarchical_tokens_enabled", False)),
        "coarse_patch_size": int(getattr(cfg, "coarse_patch_size", 4)),
        "x0_aux_weight": float(getattr(cfg, "x0_aux_weight", 0.0)),
    }
    text_config = {
        "encoders": text_metadata.get("encoders", cfg_dict.get("text", {}).get("encoders", [])),
        "text_dim": int(cfg.text_dim),
        "pooled_dim": int(cfg.pooled_dim),
        "max_length_total": int(
            sum(int(item.get("max_length", 0)) for item in text_metadata.get("encoders", []))
        ),
    }
    vae_config = {
        "pretrained": str(cfg.vae_pretrained),
        "scaling_factor": float(cfg.vae_scaling_factor),
    }
    flow_config = {
        "timestep_sampling": str(cfg.flow_timestep_sampling),
        "logit_mean": float(cfg.flow_logit_mean),
        "logit_std": float(cfg.flow_logit_std),
        "loss_weighting": str(cfg.flow_loss_weighting),
        "timestep_shift": float(getattr(cfg, "flow_timestep_shift", 1.0)),
        "train_t_min": float(cfg.flow_train_t_min),
        "train_t_max": float(cfg.flow_train_t_max),
    }
    return {
        "architecture": "mmdit_rf",
        "objective": "rectified_flow",
        "prediction_type": "flow_velocity",
        "model_config": model_config,
        "text_config": text_config,
        "vae_config": vae_config,
        "flow_config": flow_config,
        "train_config_hash": stable_config_hash(cfg_dict),
        "dataset_hash": str(dataset_hash),
        "step": int(step),
    }


def validate_checkpoint_metadata(metadata: dict[str, Any]) -> None:
    missing = [field for field in _REQUIRED_METADATA_FIELDS if field not in metadata]
    if missing:
        raise RuntimeError("Checkpoint metadata missing required field(s): " + ", ".join(missing))


def read_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    ckpt = _torch_load(str(path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint payload must be a dict.")
    metadata = _ckpt_metadata(ckpt)
    validate_checkpoint_metadata(metadata)
    return metadata


def _ck_value(ckpt: dict, key: str) -> Any:
    meta = _ckpt_metadata(ckpt)
    if key in _TOP_LEVEL_COMPAT_FIELDS:
        return meta.get(key, ckpt.get(key, ckpt.get("cfg", {}).get(key)))
    if key == "vae_scaling_factor":
        return meta.get("vae_config", {}).get("scaling_factor", ckpt.get("vae", {}).get("scaling_factor"))
    if key == "text_dim":
        return meta.get("text_config", {}).get("text_dim", ckpt.get("text_dim", ckpt.get("cfg", {}).get("text_dim")))
    if key == "pooled_dim":
        return meta.get("text_config", {}).get("pooled_dim", ckpt.get("pooled_dim", ckpt.get("cfg", {}).get("pooled_dim")))
    return meta.get("model_config", {}).get(key, ckpt.get(key, ckpt.get("cfg", {}).get(key)))


def _cfg_value(cfg: dict, key: str) -> Any:
    if key in _TOP_LEVEL_COMPAT_FIELDS:
        return cfg.get(key)
    if key == "vae_scaling_factor":
        return cfg.get("vae_scaling_factor", cfg.get("vae", {}).get("scaling_factor"))
    if key in {"text_dim", "pooled_dim"}:
        return cfg.get(key, cfg.get("text", {}).get(key))

    model = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    text = cfg.get("text", {}) if isinstance(cfg.get("text", {}), dict) else {}
    control = cfg.get("control", {}) if isinstance(cfg.get("control", {}), dict) else {}
    rope = model.get("rope", {}) if isinstance(model.get("rope", {}), dict) else {}
    hierarchical = model.get("hierarchical", {}) if isinstance(model.get("hierarchical", {}), dict) else {}
    resampler = text.get("resampler", {}) if isinstance(text.get("resampler", {}), dict) else {}
    loss = cfg.get("loss", {}) if isinstance(cfg.get("loss", {}), dict) else {}

    nested_stage_a = {
        "rope_scaling": rope.get("scaling"),
        "rope_base_grid_hw": rope.get("base_grid"),
        "rope_theta": rope.get("theta"),
        "hierarchical_tokens_enabled": hierarchical.get("enabled"),
        "coarse_patch_size": hierarchical.get("coarse_patch_size"),
        "text_resampler_enabled": resampler.get("enabled"),
        "text_resampler_num_tokens": resampler.get("num_tokens"),
        "text_resampler_depth": resampler.get("depth"),
        "text_resampler_mlp_ratio": resampler.get("mlp_ratio"),
        "x0_aux_weight": loss.get("x0_aux_weight"),
        "control_type_embed": control.get("type_embed"),
        "control_adapter": control.get("adapter"),
        "control_adapter_ratio": control.get("adapter_ratio"),
    }
    if key in nested_stage_a and nested_stage_a[key] is not None:
        return nested_stage_a[key]
    return cfg.get(key, model.get(key))



def _normalize_compat_value(key: str, value: Any) -> Any:
    if key == "rope_base_grid_hw" and value is not None:
        seq = list(value)
        if len(seq) != 2:
            return value
        return (int(seq[0]), int(seq[1]))
    if key in {
        "rope_theta",
        "text_resampler_mlp_ratio",
        "mlp_ratio",
        "control_adapter_ratio",
        "vae_scaling_factor",
    } and value is not None:
        return float(value)
    return value

def _raise_mismatch(key: str, ck_value: Any, cfg_value: Any) -> None:
    raise RuntimeError(
        f"Checkpoint incompatible: {key} mismatch.\n"
        f"checkpoint: {ck_value}\n"
        f"current config: {cfg_value}"
    )


def validate_mmdit_checkpoint_compatibility(ckpt: dict, cfg: dict) -> None:
    for key in (*_TOP_LEVEL_COMPAT_FIELDS, *_MODEL_COMPAT_FIELDS, "text_dim", "pooled_dim", "vae_scaling_factor"):
        ck_value = _ck_value(ckpt, key)
        cfg_value = _cfg_value(cfg, key)
        ck_cmp = _normalize_compat_value(key, ck_value)
        cfg_cmp = _normalize_compat_value(key, cfg_value)
        if cfg_cmp is not None and ck_cmp is not None and ck_cmp != cfg_cmp:
            _raise_mismatch(key, ck_value, cfg_value)

    ck_meta = _ckpt_metadata(ckpt)
    ck_text = ck_meta.get("text_config", {}).get(
        "encoders",
        ckpt.get("text_encoders", ckpt.get("cfg", {}).get("text", {}).get("encoders", [])),
    )
    cfg_text = _config_text_encoders(cfg)
    if cfg_text and ck_text and _normalize_text_encoders(list(ck_text)) != _normalize_text_encoders(list(cfg_text)):
        _raise_mismatch("text_encoders differ", _normalize_text_encoders(list(ck_text)), _normalize_text_encoders(list(cfg_text)))

    ck_text_max = ck_meta.get("text_config", {}).get("max_length_total")
    cfg_text_max = sum(int(item.get("max_length", 0)) for item in cfg_text) if cfg_text else None
    if cfg_text_max is not None and ck_text_max not in {None, 0} and int(ck_text_max) != int(cfg_text_max):
        _raise_mismatch("text max lengths", ck_text_max, cfg_text_max)
