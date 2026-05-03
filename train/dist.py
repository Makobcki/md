from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _dist_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _dist_rank() -> int:
    if _dist_is_initialized():
        return torch.distributed.get_rank()
    return 0


def _dist_world_size() -> int:
    if _dist_is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _dist_all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if _dist_is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor


@dataclass
class DistributedContext:
    """Small adapter around optional Accelerate/DDP state.

    The default context is deliberately a no-op so single-process training stays
    byte-for-byte close to the original path. When backend="accelerate", objects
    are prepared through ``Accelerator.prepare`` and rank-gated side effects such
    as checkpoint/event writes can use the same interface without importing
    Accelerate in most modules.
    """

    backend: str = "none"
    accelerator: Any | None = None
    device: torch.device | None = None
    save_on_rank0_only: bool = True
    aggregate_metrics: bool = True

    @property
    def is_distributed(self) -> bool:
        return self.backend != "none" or _dist_world_size() > 1

    @property
    def is_main_process(self) -> bool:
        if self.accelerator is not None:
            return bool(getattr(self.accelerator, "is_main_process", True))
        return _dist_rank() == 0

    @property
    def rank(self) -> int:
        if self.accelerator is not None:
            return int(getattr(self.accelerator, "process_index", 0))
        return _dist_rank()

    @property
    def world_size(self) -> int:
        if self.accelerator is not None:
            return int(getattr(self.accelerator, "num_processes", 1))
        return _dist_world_size()

    def prepare(self, *objects: Any) -> tuple[Any, ...]:
        if self.accelerator is not None:
            prepared = self.accelerator.prepare(*objects)
            if not isinstance(prepared, tuple):
                return (prepared,)
            return prepared
        return objects

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(model)
        if hasattr(model, "_orig_mod"):
            return getattr(model, "_orig_mod")
        if hasattr(model, "module"):
            return getattr(model, "module")
        return model

    def wait_for_everyone(self) -> None:
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        elif _dist_is_initialized():
            torch.distributed.barrier()

    def broadcast_object(self, value: Any, *, src: int = 0) -> Any:
        if self.world_size <= 1:
            return value
        objects = [value]
        if self.accelerator is not None and hasattr(self.accelerator, "broadcast_object_list"):
            self.accelerator.broadcast_object_list(objects, from_process=src)
            return objects[0]
        if _dist_is_initialized():
            torch.distributed.broadcast_object_list(objects, src=src)
            return objects[0]
        return value

    def reduce_mean_float(self, value: float, *, device: torch.device | None = None) -> float:
        if not self.aggregate_metrics:
            return float(value)
        world = self.world_size
        if world <= 1:
            return float(value)
        dev = device or self.device or torch.device("cpu")
        tensor = torch.tensor(float(value), device=dev, dtype=torch.float32)
        if self.accelerator is not None:
            tensor = self.accelerator.reduce(tensor, reduction="mean")
        elif _dist_is_initialized():
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            tensor /= float(world)
        return float(tensor.detach().cpu())

    def should_write(self) -> bool:
        return (not self.save_on_rank0_only) or self.is_main_process


def create_distributed_context(cfg: Any, *, device: torch.device | None = None) -> DistributedContext:
    backend = str(getattr(cfg, "distributed_backend", "none") or "none")
    if backend == "none":
        return DistributedContext(
            backend="none",
            device=device,
            save_on_rank0_only=bool(getattr(cfg, "save_on_rank0_only", True)),
            aggregate_metrics=bool(getattr(cfg, "distributed_metrics_aggregation", True)),
        )
    if backend != "accelerate":
        raise RuntimeError(f"Unsupported distributed_backend: {backend}")
    if bool(getattr(cfg, "fsdp_enabled", False)):
        raise RuntimeError(
            "FSDP is planned but intentionally not enabled in this trainer yet. "
            "Use distributed_backend=accelerate with fsdp.enabled=false, or keep the FSDP template as documentation only."
        )
    try:
        from accelerate import Accelerator
    except ImportError as exc:
        raise RuntimeError("distributed_backend=accelerate requires the accelerate package.") from exc

    # This trainer keeps its explicit AMP/GradScaler path so Accelerate is used
    # for process placement, DDP wrapping, dataloader sharding, and rank-aware IO.
    accelerator = Accelerator(cpu=device is not None and device.type == "cpu", mixed_precision="no")
    return DistributedContext(
        backend="accelerate",
        accelerator=accelerator,
        device=accelerator.device,
        save_on_rank0_only=bool(getattr(cfg, "save_on_rank0_only", True)),
        aggregate_metrics=bool(getattr(cfg, "distributed_metrics_aggregation", True)),
    )
