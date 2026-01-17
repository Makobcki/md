from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerfConfig:
    tf32: bool
    cudnn_benchmark: bool
    channels_last: bool
    enable_flash_sdp: bool
    enable_mem_efficient_sdp: bool
    enable_math_sdp: bool
