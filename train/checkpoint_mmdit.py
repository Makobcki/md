from __future__ import annotations


def validate_mmdit_checkpoint_compatibility(ckpt: dict, cfg: dict) -> None:
    ck_cfg = ckpt.get("cfg", {})
    checks = (
        "architecture",
        "hidden_dim",
        "depth",
        "latent_patch_size",
        "latent_channels",
        "objective",
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
    if cfg_text and ck_text and ck_text != cfg_text:
        mismatches.append("text_encoders differ")
    if mismatches:
        raise RuntimeError("mmdit checkpoint config mismatch: " + "; ".join(mismatches))

