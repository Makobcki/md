from __future__ import annotations

import torch


def _dist_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _dist_rank() -> int:
    if _dist_is_initialized():
        return torch.distributed.get_rank()
    return 0


def _dist_all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if _dist_is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor
