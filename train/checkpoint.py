from __future__ import annotations

import os
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
    "link_checkpoint_alias",
    "_prune_checkpoints",
]


def _link_one(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp_alias")
    try:
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
    except FileNotFoundError:
        pass
    try:
        os.link(src, tmp)
        os.replace(tmp, dst)
        return
    except OSError:
        try:
            if tmp.exists() or tmp.is_symlink():
                tmp.unlink()
        except FileNotFoundError:
            pass
    try:
        rel_src = os.path.relpath(src, start=dst.parent)
        tmp.symlink_to(rel_src)
        os.replace(tmp, dst)
    finally:
        try:
            if tmp.exists() and not dst.exists():
                tmp.unlink()
        except FileNotFoundError:
            pass


def link_checkpoint_alias(src: Path, dst: Path) -> None:
    """Create/update a legacy checkpoint alias without duplicating checkpoint bytes.

    Prefer a hardlink (same inode, no extra data blocks), then a relative symlink.
    If neither is supported by the filesystem, leave the alias absent instead of
    silently making a full checkpoint copy. Metadata sidecars are aliased too.
    """
    src = Path(src)
    dst = Path(dst)
    _link_one(src, dst)
    src_meta = src.with_suffix(src.suffix + ".metadata.json")
    if src_meta.exists():
        _link_one(src_meta, dst.with_suffix(dst.suffix + ".metadata.json"))


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
            for item in (p, p.with_suffix(p.suffix + ".metadata.json")):
                try:
                    item.unlink()
                except FileNotFoundError:
                    continue
