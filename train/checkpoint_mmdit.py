from __future__ import annotations


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


def validate_mmdit_checkpoint_compatibility(ckpt: dict, cfg: dict) -> None:
    ck_cfg = ckpt.get("cfg", {})
    checks = (
        "architecture",
        "objective",
        "latent_channels",
        "latent_patch_size",
        "hidden_dim",
        "depth",
        "num_heads",
        "double_stream_blocks",
        "single_stream_blocks",
        "text_dim",
        "pooled_dim",
        "vae_scaling_factor",
    )
    mismatches: list[str] = []
    for key in checks:
        ck_value = ckpt.get(key, ck_cfg.get(key))
        cfg_value = cfg.get(key)
        if cfg_value is not None and ck_value is not None and ck_value != cfg_value:
            mismatches.append(f"{key}: ckpt={ck_value!r}, cfg={cfg_value!r}")
    ck_text = ckpt.get("text_encoders", ck_cfg.get("text", {}).get("encoders", []))
    cfg_text = cfg.get("text_encoders", cfg.get("text", {}).get("encoders", []))
    if cfg_text and ck_text and _normalize_text_encoders(ck_text) != _normalize_text_encoders(cfg_text):
        mismatches.append("text_encoders differ")
    if mismatches:
        raise RuntimeError("mmdit checkpoint config mismatch: " + "; ".join(mismatches))
