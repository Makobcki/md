from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextEncoderConfig:
    name: str
    model_name: str
    max_length: int
    trainable: bool = False
    cache: bool = True

