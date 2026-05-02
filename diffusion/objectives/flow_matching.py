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


def sample_timestep(
    batch_size: int,
    *,
    mode: Literal["uniform", "logit_normal"] = "logit_normal",
    device: torch.device,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> torch.Tensor:
    if mode == "uniform":
        t = torch.rand(batch_size, device=device)
    elif mode == "logit_normal":
        z = torch.randn(batch_size, device=device) * float(logit_std) + float(logit_mean)
        t = torch.sigmoid(z)
    else:
        raise ValueError(f"Unsupported timestep sampling mode: {mode}")
    return t.clamp(float(t_min), float(t_max))


class RectifiedFlowObjective:
    def __init__(
        self,
        *,
        timestep_sampling: Literal["uniform", "logit_normal"] = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        train_t_min: float = 0.0,
        train_t_max: float = 1.0,
        loss_weighting: str = "none",
    ) -> None:
        self.timestep_sampling = timestep_sampling
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)
        self.train_t_min = float(train_t_min)
        self.train_t_max = float(train_t_max)
        self.loss_weighting = str(loss_weighting)

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
        )
        t_view = t.view(b, 1, 1, 1).to(dtype=x0.dtype)
        xt = (1.0 - t_view) * x0 + t_view * eps
        target = eps - x0
        weight = torch.ones(b, device=x0.device, dtype=torch.float32)
        return TrainingTuple(xt=xt, t=t, target=target, weight=weight)


def rectified_flow_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    per = (pred - target.to(dtype=pred.dtype)).pow(2).mean(dim=[1, 2, 3])
    return (per * weight.to(device=pred.device, dtype=per.dtype)).mean()

