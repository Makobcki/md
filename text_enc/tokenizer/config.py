from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextConfig:
    vocab_path: str
    merges_path: str
    max_len: int = 128
    lowercase: bool = True
    strip_punct: bool = True
