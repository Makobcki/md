from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

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
    "double_stream_blocks",
    "single_stream_blocks",
    "pos_embed",
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
        "double_stream_blocks": int(cfg.double_stream_blocks),
        "single_stream_blocks": int(cfg.single_stream_blocks),
        "pos_embed": str(cfg.pos_embed),
        "text_dim": int(cfg.text_dim),
        "pooled_dim": int(cfg.pooled_dim),
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
    ckpt = torch.load(str(path), map_location="cpu")
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
    return cfg.get(key, cfg.get("model", {}).get(key))


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
        if cfg_value is not None and ck_value is not None and ck_value != cfg_value:
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
