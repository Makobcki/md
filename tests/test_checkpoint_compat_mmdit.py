from __future__ import annotations

import pytest

from train.checkpoint_mmdit import validate_mmdit_checkpoint_compatibility


def test_mmdit_checkpoint_compatibility_detects_mismatch() -> None:
    ckpt = {"architecture": "mmdit_rf", "cfg": {"hidden_dim": 64, "depth": 2}}
    validate_mmdit_checkpoint_compatibility(ckpt, {"architecture": "mmdit_rf", "hidden_dim": 64})
    with pytest.raises(RuntimeError):
        validate_mmdit_checkpoint_compatibility(ckpt, {"architecture": "mmdit_rf", "hidden_dim": 128})

