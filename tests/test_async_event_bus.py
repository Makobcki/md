from __future__ import annotations

import json
from pathlib import Path

from diffusion.io.events import AsyncEventBus, JsonlFileSink


def test_async_event_bus_flushes_jsonl_events(tmp_path: Path) -> None:
    path = tmp_path / "metrics" / "train_metrics.jsonl"
    bus = AsyncEventBus([JsonlFileSink(path, event_types=["progress", "train"])])
    bus.emit({"type": "progress", "step": 0})
    for step in range(1, 4):
        bus.emit({"type": "train", "step": step, "loss": 1.0 / step})
    bus.emit({"type": "log", "message": "not a metric"})
    bus.close()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["step"] for row in rows] == [0, 1, 2, 3]
    assert rows[0]["type"] == "progress"
    assert all(row["type"] == "train" for row in rows[1:])
