from __future__ import annotations

import torch

from config.train import TrainConfig
from diffusion.perf import PerfConfig, configure_performance


def test_nested_performance_config_maps_to_sdp_flags() -> None:
    cfg = TrainConfig.from_dict(
        {
            "performance": {
                "flash_sdp": False,
                "mem_efficient_sdp": True,
                "math_sdp": True,
                "tf32": False,
                "channels_last": False,
            }
        }
    )
    assert cfg.enable_flash_sdp is False
    assert cfg.enable_mem_efficient_sdp is True
    assert cfg.enable_math_sdp is True
    assert cfg.tf32 is False
    assert cfg.channels_last is False


def test_configure_performance_has_cpu_fallback_without_enabling_cuda_sdp() -> None:
    active = configure_performance(
        PerfConfig(
            tf32=False,
            cudnn_benchmark=False,
            channels_last=False,
            enable_flash_sdp=True,
            enable_mem_efficient_sdp=True,
            enable_math_sdp=False,
        ),
        torch.device("cpu"),
    )
    assert active["tf32"] is False
    assert active["channels_last"] is False
    assert active["sdp_flash"] is False
    assert active["sdp_mem_efficient"] is False
    assert active["sdp_math"] is False
