from __future__ import annotations

import pytest

from config.resolver import resolve_config


def test_target_validation_rejects_wrong_entrypoint() -> None:
    with pytest.raises(RuntimeError, match="Config target mismatch"):
        resolve_config("configs/webui.kdl", expected_target="train")
