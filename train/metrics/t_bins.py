from __future__ import annotations

from typing import Mapping

import torch


def t_bin_key(prefix: str, idx: int, bins: int) -> str:
    if bins <= 0:
        raise ValueError("bins must be positive.")
    if idx < 0 or idx >= bins:
        raise ValueError(f"idx must be in [0, {bins}), got {idx}.")
    # The public events.jsonl schema names bins by ordinal interval, not by
    # percentage endpoints: for 10 bins this yields loss_t_bin_00_01,
    # loss_t_bin_01_02, ..., loss_t_bin_09_10.  This matches the documented
    # [0.0, 0.1), ..., [0.9, 1.0] layout while keeping keys short.
    return f"{prefix}_{idx:02d}_{idx + 1:02d}"


def loss_by_t_bins(
    per_sample_loss: torch.Tensor,
    t: torch.Tensor,
    *,
    bins: int = 10,
    prefix: str = "loss_t_bin",
) -> dict[str, float]:
    """Average per-sample loss inside uniform timestep bins over [0, 1].

    Bins are half-open except the last one, which includes t=1.0:
    [0.0, 0.1), ..., [0.9, 1.0] for bins=10.
    Empty bins are omitted so callers can safely merge the returned mapping into
    event dictionaries without serializing NaN/None values.
    """

    if bins <= 0:
        raise ValueError("bins must be positive.")
    losses = per_sample_loss.detach().float().reshape(-1)
    timesteps = t.detach().float().reshape(-1).to(device=losses.device)
    if losses.numel() != timesteps.numel():
        raise ValueError(
            "per_sample_loss and t must have the same number of elements; "
            f"got {losses.numel()} and {timesteps.numel()}."
        )
    if losses.numel() == 0:
        return {}

    out: dict[str, float] = {}
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        if idx == bins - 1:
            mask = (timesteps >= lo) & (timesteps <= hi)
        else:
            mask = (timesteps >= lo) & (timesteps < hi)
        if mask.any():
            out[t_bin_key(prefix, idx, bins)] = float(losses[mask].mean().cpu())
    return out


def required_t_bin_keys(*, bins: int = 10, prefix: str = "loss_t_bin") -> list[str]:
    return [t_bin_key(prefix, idx, bins) for idx in range(bins)]
