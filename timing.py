from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class _TimingBucket:
    count: int = 0
    total_sec: float = 0.0
    min_sec: float = float("inf")
    max_sec: float = 0.0

    def add(self, value_sec: float) -> None:
        self.count += 1
        self.total_sec += value_sec
        self.min_sec = min(self.min_sec, value_sec)
        self.max_sec = max(self.max_sec, value_sec)

    def summary(self) -> dict:
        if self.count == 0:
            return {"count": 0, "total_sec": 0.0, "avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
        avg_ms = (self.total_sec / self.count) * 1000.0
        return {
            "count": self.count,
            "total_sec": self.total_sec,
            "avg_ms": avg_ms,
            "min_ms": self.min_sec * 1000.0,
            "max_ms": self.max_sec * 1000.0,
        }


class _TimingStats:
    def __init__(self, *, use_cuda_events: bool, gpu_sections: Optional[set[str]] = None) -> None:
        self.use_cuda_events = bool(use_cuda_events)
        self.gpu_sections = gpu_sections or set()
        self._cpu: dict[str, _TimingBucket] = {}
        self._gpu_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}

    def section(self, name: str):
        return _SectionTimer(self, name)

    def add_cpu(self, name: str, elapsed_sec: float) -> None:
        bucket = self._cpu.setdefault(name, _TimingBucket())
        bucket.add(elapsed_sec)

    def add_gpu_event(self, name: str, start: torch.cuda.Event, end: torch.cuda.Event) -> None:
        self._gpu_events.setdefault(name, []).append((start, end))

    def report(self, *, reset: bool = True) -> dict:
        stats: dict[str, dict] = {}
        for name, bucket in self._cpu.items():
            stats[name] = bucket.summary()

        if self._gpu_events:
            torch.cuda.synchronize()
            for name, pairs in self._gpu_events.items():
                bucket = _TimingBucket()
                for start, end in pairs:
                    elapsed_ms = float(start.elapsed_time(end))
                    bucket.add(elapsed_ms / 1000.0)
                stats[f"{name}_gpu"] = bucket.summary()
        if reset:
            self._cpu = {}
            self._gpu_events = {}
        return stats


class _SectionTimer:
    def __init__(self, stats: _TimingStats, name: str) -> None:
        self.stats = stats
        self.name = name
        self.elapsed_sec: float = 0.0
        self._start_cpu: Optional[float] = None
        self._start_event: Optional[torch.cuda.Event] = None
        self._end_event: Optional[torch.cuda.Event] = None

    def __enter__(self) -> "_SectionTimer":
        self._start_cpu = time.perf_counter()
        if self.stats.use_cuda_events and self.name in self.stats.gpu_sections:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._start_cpu is not None:
            self.elapsed_sec = time.perf_counter() - self._start_cpu
            self.stats.add_cpu(self.name, self.elapsed_sec)
        if self._start_event is not None and self._end_event is not None:
            self._end_event.record()
            self.stats.add_gpu_event(self.name, self._start_event, self._end_event)
