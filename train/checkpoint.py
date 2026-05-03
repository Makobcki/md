from __future__ import annotations

from pathlib import Path

from diffusion.io.ckpt import (
    load_ckpt,
    normalize_state_dict_for_keys,
    normalize_state_dict_for_model,
    resolve_resume_path,
    save_ckpt,
)

__all__ = [
    "load_ckpt",
    "normalize_state_dict_for_keys",
    "normalize_state_dict_for_model",
    "resolve_resume_path",
    "save_ckpt",
    "_prune_checkpoints",
]


def _prune_checkpoints(out_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    patterns = (
        "ckpt_[0-9][0-9][0-9][0-9][0-9][0-9][0-9].pt",
        "step_[0-9][0-9][0-9][0-9][0-9][0-9].pt",
    )
    for pattern in patterns:
        ckpts = sorted(out_dir.glob(pattern))
        to_remove = ckpts[:-keep_last]
        for p in to_remove:
            try:
                p.unlink()
            except FileNotFoundError:
                continue
