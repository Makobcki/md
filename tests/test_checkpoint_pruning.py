from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from train.checkpoint import _prune_checkpoints


def test_prune_checkpoints_keeps_special_checkpoints(tmp_path: Path) -> None:
    regular = [
        tmp_path / "ckpt_0000001.pt",
        tmp_path / "ckpt_0000002.pt",
        tmp_path / "ckpt_0000003.pt",
    ]
    special = [
        tmp_path / "ckpt_best.pt",
        tmp_path / "ckpt_best_val.pt",
        tmp_path / "ckpt_latest.pt",
        tmp_path / "ckpt_final.pt",
        tmp_path / "ckpt_stop_0000002.pt",
    ]
    for path in regular + special:
        path.write_text(path.name, encoding="utf-8")

    _prune_checkpoints(tmp_path, keep_last=1)

    assert not regular[0].exists()
    assert not regular[1].exists()
    assert regular[2].exists()
    assert all(path.exists() for path in special)
