from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass(frozen=True)
class PerfConfig:
    tf32: bool
    cudnn_benchmark: bool
    channels_last: bool
    enable_flash_sdp: bool
    enable_mem_efficient_sdp: bool
    enable_math_sdp: bool


def configure_performance(cfg: PerfConfig, device: torch.device) -> Dict[str, bool]:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
    torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)
    torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark)

    active = {
        "tf32": bool(cfg.tf32),
        "cudnn_benchmark": bool(cfg.cudnn_benchmark),
        "channels_last": bool(cfg.channels_last),
        "sdp_flash": False,
        "sdp_mem_efficient": False,
        "sdp_math": False,
    }

    if device.type == "cuda":
        enable_flash = bool(cfg.enable_flash_sdp)
        enable_mem_efficient = bool(cfg.enable_mem_efficient_sdp)
        enable_math = bool(cfg.enable_math_sdp) or not (enable_flash or enable_mem_efficient)
        torch.backends.cuda.enable_flash_sdp(enable_flash)
        torch.backends.cuda.enable_mem_efficient_sdp(enable_mem_efficient)
        torch.backends.cuda.enable_math_sdp(enable_math)

        active.update({
            "sdp_flash": enable_flash,
            "sdp_mem_efficient": enable_mem_efficient,
            "sdp_math": enable_math,
        })

    return active
