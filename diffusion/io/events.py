from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol


class EventSink(Protocol):
    def emit(self, event: dict) -> None:
        ...


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
