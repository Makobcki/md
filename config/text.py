from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextConfig:
    vocab_path: str
    merges_path: str
    tokenizer_type: str
    max_len: int
