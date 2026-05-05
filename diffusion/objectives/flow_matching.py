from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch


@dataclass(frozen=True)
class TrainingTuple:
    xt: torch.Tensor
    t: torch.Tensor
    target: torch.Tensor
    weight: torch.Tensor


class Objective(Protocol):
    def sample_training_tuple(self, x0: torch.Tensor) -> TrainingTuple:
        ...


def apply_timestep_shift(t: torch.Tensor, shift: float) -> torch.Tensor:
    shift = float(shift)
    if shift <= 0:
        raise ValueError("timestep shift must be positive")
    if shift == 1.0:
        return t
    return (shift * t) / (1.0 + (shift - 1.0) * t)


def sample_timestep(
    batch_size: int,
    *,
    mode: Literal["uniform", "logit_normal", "shifted_logit_normal", "cosmap", "cosmap_like"] = "logit_normal",
    device: torch.device,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
    shift: float = 1.0,
) -> torch.Tensor:
    if mode == "uniform":
        t = torch.rand(batch_size, device=device)
    elif mode in {"logit_normal", "shifted_logit_normal"}:
        z = torch.randn(batch_size, device=device) * float(logit_std) + float(logit_mean)
        t = torch.sigmoid(z)
        if mode == "shifted_logit_normal":
            t = apply_timestep_shift(t, shift)
    elif mode in {"cosmap", "cosmap_like"}:
        u = torch.rand(batch_size, device=device)
        t = 1.0 - torch.cos(u * torch.pi * 0.5)
    else:
        raise ValueError(f"Unsupported timestep sampling mode: {mode}")
    return t.clamp(float(t_min), float(t_max))


class RectifiedFlowObjective:
    def __init__(
        self,
        *,
        timestep_sampling: Literal["uniform", "logit_normal", "shifted_logit_normal", "cosmap", "cosmap_like"] = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        train_t_min: float = 0.0,
        train_t_max: float = 1.0,
        loss_weighting: str = "none",
        timestep_shift: float = 1.0,
    ) -> None:
        allowed_sampling = {"uniform", "logit_normal", "shifted_logit_normal", "cosmap", "cosmap_like"}
        if str(timestep_sampling) not in allowed_sampling:
            allowed = ", ".join(sorted(allowed_sampling))
            raise ValueError(f"Unsupported timestep sampling mode: {timestep_sampling}. Allowed: {allowed}.")
        if float(logit_std) <= 0:
            raise ValueError("logit_std must be positive.")
        if not (0.0 <= float(train_t_min) <= 1.0 and 0.0 <= float(train_t_max) <= 1.0):
            raise ValueError("train_t_min and train_t_max must be in [0, 1].")
        if float(train_t_min) > float(train_t_max):
            raise ValueError("train_t_min must be <= train_t_max.")
        if float(timestep_shift) <= 0:
            raise ValueError("timestep_shift must be positive.")
        if str(loss_weighting) not in {"none"}:
            raise ValueError("loss_weighting must be 'none'.")
        self.timestep_sampling = str(timestep_sampling)
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)
        self.train_t_min = float(train_t_min)
        self.train_t_max = float(train_t_max)
        self.loss_weighting = str(loss_weighting)
        self.timestep_shift = float(timestep_shift)

    @classmethod
    def from_config(cls, cfg: dict) -> "RectifiedFlowObjective":
        flow = cfg.get("flow", cfg)
        return cls(
            timestep_sampling=str(flow.get("timestep_sampling", "logit_normal")),
            logit_mean=float(flow.get("logit_mean", 0.0)),
            logit_std=float(flow.get("logit_std", 1.0)),
            train_t_min=float(flow.get("train_t_min", 0.0)),
            train_t_max=float(flow.get("train_t_max", 1.0)),
            loss_weighting=str(flow.get("loss_weighting", "none")),
            timestep_shift=float(flow.get("timestep_shift", flow.get("shift", 1.0))),
        )

    def sample_training_tuple(self, x0: torch.Tensor) -> TrainingTuple:
        b = x0.shape[0]
        eps = torch.randn_like(x0)
        t = sample_timestep(
            b,
            mode=self.timestep_sampling,
            device=x0.device,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            t_min=self.train_t_min,
            t_max=self.train_t_max,
            shift=self.timestep_shift,
        )
        t_view = t.view(b, 1, 1, 1).to(dtype=x0.dtype)
        xt = (1.0 - t_view) * x0 + t_view * eps
        target = eps - x0
        weight = torch.ones(b, device=x0.device, dtype=torch.float32)
        return TrainingTuple(xt=xt, t=t, target=target, weight=weight)


def rectified_flow_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    err = (pred.float() - target.to(device=pred.device, dtype=torch.float32)).pow(2)
    per = err.mean(dim=[1, 2, 3])
    if mask is not None:
        m = mask.to(device=pred.device, dtype=torch.float32)
        if m.dim() == 3:
            m = m.unsqueeze(1)
        denom = (m.sum(dim=[1, 2, 3]) * pred.shape[1]).clamp_min(1.0)
        masked = (err * m).sum(dim=[1, 2, 3]) / denom
        has_mask = m.sum(dim=[1, 2, 3]) > 0
        per = torch.where(has_mask, masked, per)
    return (per * weight.to(device=pred.device, dtype=per.dtype)).mean()

