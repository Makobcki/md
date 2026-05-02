from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import TextIO

import torch


_MEMORY_UNITS = {
    "b": 1,
    "byte": 1,
    "bytes": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}
_MEMORY_VALUE_RE = r"([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?i?B|bytes?)"


@dataclass(frozen=True)
class OomMemoryInfo:
    requested_bytes: int | None = None
    total_bytes: int | None = None
    free_bytes: int | None = None
    process_used_bytes: int | None = None
    pytorch_allocated_bytes: int | None = None
    pytorch_reserved_unallocated_bytes: int | None = None

    @property
    def current_used_bytes(self) -> int | None:
        if self.total_bytes is None or self.free_bytes is None:
            return self.process_used_bytes
        return max(self.total_bytes - self.free_bytes, 0)

    @property
    def required_total_bytes(self) -> int | None:
        if self.current_used_bytes is None or self.requested_bytes is None:
            return None
        return self.current_used_bytes + self.requested_bytes

    @property
    def shortfall_bytes(self) -> int | None:
        if self.required_total_bytes is None or self.total_bytes is None:
            return None
        return max(self.required_total_bytes - self.total_bytes, 0)


def is_torch_oom_error(exc: BaseException) -> bool:
    return find_torch_oom_error(exc) is not None


def find_torch_oom_error(exc: BaseException) -> BaseException | None:
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, torch.OutOfMemoryError):
            return current
        message = str(current).lower()
        if "out of memory" in message and ("cuda" in message or "mps" in message or "torch" in message):
            return current
        if current.__cause__ is not None:
            stack.append(current.__cause__)
        if current.__context__ is not None:
            stack.append(current.__context__)
    return None


def _parse_memory_value(text: str) -> int:
    match = re.fullmatch(_MEMORY_VALUE_RE, text.strip(), flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"invalid memory value: {text!r}")
    value = float(match.group(1))
    unit = match.group(2).lower()
    return int(round(value * _MEMORY_UNITS[unit]))


def parse_torch_oom_message(message: str) -> OomMemoryInfo:
    def _search(pattern: str) -> int | None:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match is None:
            return None
        return _parse_memory_value(match.group(1))

    requested = _search(r"Tried to allocate\s+(" + _MEMORY_VALUE_RE + r")")
    total_free = re.search(
        r"total capacity of\s+(" + _MEMORY_VALUE_RE + r")\s+of which\s+(" + _MEMORY_VALUE_RE + r")\s+is free",
        message,
        flags=re.IGNORECASE,
    )
    if total_free is not None:
        total = _parse_memory_value(total_free.group(1))
        free = _parse_memory_value(total_free.group(4))
    else:
        total = None
        free = None

    process_used = _search(r"this process has\s+(" + _MEMORY_VALUE_RE + r")\s+memory in use")
    allocated = _search(r"allocated memory\s+(" + _MEMORY_VALUE_RE + r")\s+is allocated by PyTorch")
    reserved = _search(r"and\s+(" + _MEMORY_VALUE_RE + r")\s+is reserved by PyTorch but unallocated")

    return OomMemoryInfo(
        requested_bytes=requested,
        total_bytes=total,
        free_bytes=free,
        process_used_bytes=process_used,
        pytorch_allocated_bytes=allocated,
        pytorch_reserved_unallocated_bytes=reserved,
    )


def format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(value) < 1024.0 or unit == "GiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.2f} GiB"


def format_torch_oom_message(exc: BaseException, *, context: str | None = None) -> str:
    oom_exc = find_torch_oom_error(exc) or exc
    info = parse_torch_oom_message(str(oom_exc))
    title = "[OOM] CUDA out of memory" if context is None else f"[OOM] CUDA out of memory during {context}"
    lines = [title]
    if info.requested_bytes is not None:
        lines.append(f"requested allocation: {format_bytes(info.requested_bytes)}")
    if info.required_total_bytes is not None:
        lines.append(f"estimated required VRAM: {format_bytes(info.required_total_bytes)}")
    if info.shortfall_bytes is not None:
        lines.append(f"estimated shortfall: {format_bytes(info.shortfall_bytes)}")
    if info.total_bytes is not None or info.free_bytes is not None:
        lines.append(f"device memory: total={format_bytes(info.total_bytes)}, free={format_bytes(info.free_bytes)}")
    if info.process_used_bytes is not None:
        lines.append(f"process memory in use: {format_bytes(info.process_used_bytes)}")
    if info.pytorch_allocated_bytes is not None or info.pytorch_reserved_unallocated_bytes is not None:
        lines.append(
            "pytorch memory: "
            f"allocated={format_bytes(info.pytorch_allocated_bytes)}, "
            f"reserved_unallocated={format_bytes(info.pytorch_reserved_unallocated_bytes)}"
        )
    if info.shortfall_bytes == 0 and info.requested_bytes is not None:
        lines.append("capacity shortfall is 0; fragmentation or non-PyTorch reservations may be the cause")
    return "\n".join(lines)


def print_torch_oom(exc: BaseException, *, context: str | None = None, file: TextIO | None = None) -> None:
    print(format_torch_oom_message(exc, context=context), file=file or sys.stderr, flush=True)
