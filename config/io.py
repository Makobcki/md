from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IOConfig:
    out_dir: str
    ckpt_keep_last: int
    resume_ckpt: str
