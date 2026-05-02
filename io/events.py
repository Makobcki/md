from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol


class EventSink(Protocol):
    def emit(self, event: dict) -> None:
        ...


def _format_number(value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    abs_value = abs(value)
    if value == 0:
        return "0"
    if abs_value >= 1000:
        return f"{value:.0f}"
    if abs_value >= 100:
        return f"{value:.1f}"
    if abs_value >= 1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.4g}"


def _format_extra_fields(event: dict, skip: set[str]) -> str:
    parts = []
    for key, value in event.items():
        if key in skip or value is None:
            continue
        if isinstance(value, (dict, list, tuple)):
            continue
        parts.append(f"{key}={_format_number(value)}")
    return " ".join(parts)


def format_event_line(event: dict) -> str:
    event_type = event.get("type")
    if event_type == "log":
        step = event.get("step")
        prefix = f"step={step} " if step is not None else ""
        message = str(event.get("message", ""))
        extras = _format_extra_fields(event, {"type", "message", "step"})
        return f"{prefix}{message}{(' ' + extras) if extras else ''}".strip()

    if event_type == "metric":
        fields = []
        max_steps = event.get("max_steps")
        for key in (
            "step",
            "processed",
            "loss",
            "val_loss",
            "lr",
            "grad_norm",
            "img_per_sec",
            "items_per_sec",
            "peak_mem_mb",
            "eta_h",
            "saved",
            "skipped",
            "errors",
            "sampler",
        ):
            value = event.get(key)
            if value is not None:
                if key in {"step", "processed"} and max_steps is not None:
                    fields.append(f"{key}={_format_number(value)}/{_format_number(max_steps)}")
                else:
                    fields.append(f"{key}={_format_number(value)}")
        return " ".join(["metric", *fields]).strip()

    if event_type == "status":
        status = event.get("status", "unknown")
        extras = _format_extra_fields(event, {"type", "status"})
        return f"status={status}{(' ' + extras) if extras else ''}"

    extras = _format_extra_fields(event, {"type"})
    return f"{event_type or 'event'} {extras}".strip()


@dataclass(frozen=True)
class StdoutJsonSink:
    # Пишет события JSON в stdout (для JobManager/WebUI).
    def emit(self, event: dict) -> None:
        print(json.dumps(event, ensure_ascii=False), flush=True)


@dataclass(frozen=True)
class JsonlFileSink:
    # Пишет события в .jsonl файл.
    path: Path
    event_types: Optional[Iterable[str]] = None

    def emit(self, event: dict) -> None:
        if self.event_types is not None and event.get("type") not in set(self.event_types):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


class EventBus:
    # Простой bus: fan-out на несколько sink-ов.
    def __init__(self, sinks: Iterable[EventSink]) -> None:
        self._sinks = list(sinks)

    def emit(self, event: dict) -> None:
        for sink in self._sinks:
            sink.emit(event)
