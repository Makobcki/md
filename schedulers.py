from __future__ import annotations

import math

import torch


def _compute_lr(
    *,
    step: int,
    base_lr: float,
    warmup_steps: int,
    decay_steps: int,
    min_lr_ratio: float,
    scheduler: str,
) -> float:
    if base_lr <= 0:
        raise RuntimeError("base_lr must be positive.")
    if warmup_steps < 0 or decay_steps < 0:
        raise RuntimeError("warmup_steps and decay_steps must be non-negative.")
    if min_lr_ratio <= 0 or min_lr_ratio > 1:
        raise RuntimeError("min_lr_ratio must be in (0, 1].")
    if scheduler not in {"cosine", "linear"}:
        raise RuntimeError(f"Unsupported lr_scheduler={scheduler}")

    min_lr = base_lr * min_lr_ratio
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    if decay_steps <= 0:
        return base_lr
    progress = min(max(step - warmup_steps, 0) / float(decay_steps), 1.0)
    if scheduler == "cosine":
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine
    return min_lr + (base_lr - min_lr) * (1.0 - progress)


def _apply_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)
