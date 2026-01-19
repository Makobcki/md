from __future__ import annotations

import torch


def v_to_x0(
    xt: torch.Tensor,
    t: torch.Tensor,
    v: torch.Tensor,
    sqrt_alpha_bar: torch.Tensor,
    sqrt_one_minus_alpha_bar: torch.Tensor,
) -> torch.Tensor:
    b = xt.shape[0]
    s1 = sqrt_alpha_bar[t].view(b, 1, 1, 1)
    s2 = sqrt_one_minus_alpha_bar[t].view(b, 1, 1, 1)
    return s1 * xt - s2 * v


def v_to_eps(
    xt: torch.Tensor,
    t: torch.Tensor,
    v: torch.Tensor,
    sqrt_alpha_bar: torch.Tensor,
    sqrt_one_minus_alpha_bar: torch.Tensor,
) -> torch.Tensor:
    b = xt.shape[0]
    s1 = sqrt_alpha_bar[t].view(b, 1, 1, 1)
    s2 = sqrt_one_minus_alpha_bar[t].view(b, 1, 1, 1)
    return s2 * xt + s1 * v


def eps_to_x0(
    xt: torch.Tensor,
    t: torch.Tensor,
    eps: torch.Tensor,
    sqrt_alpha_bar: torch.Tensor,
    sqrt_one_minus_alpha_bar: torch.Tensor,
) -> torch.Tensor:
    b = xt.shape[0]
    s1 = sqrt_alpha_bar[t].view(b, 1, 1, 1)
    s2 = sqrt_one_minus_alpha_bar[t].view(b, 1, 1, 1)
    return (xt - s2 * eps) / s1
