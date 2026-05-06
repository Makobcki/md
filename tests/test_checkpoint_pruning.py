from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from train.checkpoint import _prune_checkpoints


def test_prune_checkpoints_keeps_special_checkpoints(tmp_path: Path) -> None:
    regular = [
        tmp_path / "step_000001.pt",
        tmp_path / "step_000002.pt",
        tmp_path / "step_000003.pt",
    ]
    special = [
        tmp_path / "latest.pt",
        tmp_path / "final.pt",
        tmp_path / "best.pt",
    ]
    for path in regular + special:
        path.write_text(path.name, encoding="utf-8")

    _prune_checkpoints(tmp_path, keep_last=1)

    assert not regular[0].exists()
    assert not regular[1].exists()
    assert regular[2].exists()
    assert all(path.exists() for path in special)
