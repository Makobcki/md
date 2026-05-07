from __future__ import annotations

import hashlib
import json
from typing import Any

from model.registry import (
    build_model_capabilities,
    build_model_spec,
    build_runtime_contract,
)


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    return value if isinstance(value, dict) else {}


def _json_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tokenizer_config(cfg: dict[str, Any]) -> dict[str, Any] | None:
    model = _section(cfg, "model")
    tokenizer = _section(model, "tokenizer")
    architecture = _section(model, "architecture")
    if not tokenizer:
        return None
    out = dict(tokenizer)
    if "scale_schedule" in architecture:
        out["scale_schedule"] = [int(v) for v in architecture["scale_schedule"]]
    if "max_token_length" in architecture:
        out["max_token_length"] = int(architecture["max_token_length"])
    out["config_hash"] = _json_hash(out)
    return out


def build_model_checkpoint_metadata(
    family: str,
    cfg: dict[str, Any],
    model: Any | None = None,
    *,
    training_state: dict[str, Any] | None = None,
    optimizer_config: dict[str, Any] | None = None,
    ema_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = build_model_spec({"model": {**_section(cfg, "model"), "family": family}})
    contract = build_runtime_contract({"model": {**_section(cfg, "model"), "family": family}})
    capabilities = build_model_capabilities({"model": {**_section(cfg, "model"), "family": family}})
    model_section = _section(cfg, "model")
    architecture = _section(model_section, "architecture")
    model_config = dict(architecture)
    if model is not None and hasattr(model, "cfg"):
        model_config.update(vars(getattr(model, "cfg")))
    return {
        "checkpoint": {
            "metadata_version": 2,
            "model": {
                "family": spec.family,
                "variant": spec.variant,
                "architecture": spec.architecture,
                "objective": spec.objective,
                "prediction_type": spec.prediction_type,
                "model_config": model_config,
                "config_hash": _json_hash(model_config),
            },
            "io": {
                "input_kind": contract.input_kind,
                "output_kind": contract.output_kind,
            },
            "capabilities": capabilities.__dict__,
            "text_config": _section(cfg, "text"),
            "vae_config": _section(cfg, "vae"),
            "tokenizer_config": _tokenizer_config(cfg),
            "optimizer_config": dict(optimizer_config or {}),
            "ema_config": dict(ema_config or {}),
            "training_state": dict(training_state or {}),
        }
    }


def validate_checkpoint_compatibility(
    family: str, checkpoint: dict[str, Any], cfg: dict[str, Any]
) -> None:
    metadata = checkpoint.get("metadata") if isinstance(checkpoint, dict) else None
    if not isinstance(metadata, dict):
        if family == "mmdit":
            return
        raise RuntimeError(f"Missing checkpoint metadata v2 for model family {family}.")
    payload = metadata.get("checkpoint", metadata)
    if not isinstance(payload, dict) or int(payload.get("metadata_version", 0)) != 2:
        if family == "mmdit":
            return
        raise RuntimeError(f"Missing checkpoint metadata v2 for model family {family}.")
    model_meta = _section(payload, "model")
    actual_family = str(model_meta.get("family", ""))
    if actual_family != family:
        raise RuntimeError(f"Checkpoint family mismatch: {actual_family!r} != {family!r}.")
    expected_spec = build_model_spec({"model": {**_section(cfg, "model"), "family": family}})
    if str(model_meta.get("architecture", "")) != expected_spec.architecture:
        raise RuntimeError("Checkpoint architecture is incompatible with requested config.")
    tokenizer = payload.get("tokenizer_config")
    expected_tokenizer = _tokenizer_config(cfg)
    if expected_spec.family == "var" and tokenizer != expected_tokenizer:
        raise RuntimeError("Checkpoint tokenizer metadata is incompatible with requested config.")
