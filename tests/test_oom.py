from __future__ import annotations

import torch

from diffusion.utils.oom import (
    find_torch_oom_error,
    format_torch_oom_message,
    parse_torch_oom_message,
)


OOM_MESSAGE = (
    "CUDA out of memory. Tried to allocate 12.00 MiB. "
    "GPU 0 has a total capacity of 7.52 GiB of which 15.94 MiB is free. "
    "Including non-PyTorch memory, this process has 6.83 GiB memory in use. "
    "Of the allocated memory 6.62 GiB is allocated by PyTorch, "
    "and 40.53 MiB is reserved by PyTorch but unallocated."
)


def test_parse_torch_oom_message_estimates_required_vram() -> None:
    info = parse_torch_oom_message(OOM_MESSAGE)

    assert info.requested_bytes == 12 * 1024**2
    assert info.total_bytes == int(round(7.52 * 1024**3))
    assert info.free_bytes == int(round(15.94 * 1024**2))
    assert info.required_total_bytes == info.total_bytes - info.free_bytes + info.requested_bytes


def test_format_torch_oom_message_prints_required_memory() -> None:
    message = format_torch_oom_message(torch.OutOfMemoryError(OOM_MESSAGE), context="training")

    assert "[OOM] CUDA out of memory during training" in message
    assert "requested allocation: 12.00 MiB" in message
    assert "estimated required VRAM:" in message
    assert "device memory: total=7.52 GiB, free=15.94 MiB" in message


def test_find_torch_oom_error_walks_exception_causes() -> None:
    try:
        try:
            raise torch.OutOfMemoryError(OOM_MESSAGE)
        except torch.OutOfMemoryError as exc:
            raise RuntimeError("OOM during VAE encode") from exc
    except RuntimeError as wrapped:
        assert find_torch_oom_error(wrapped) is wrapped.__cause__
