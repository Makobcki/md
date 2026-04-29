from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Iterator

import pytest
from fastapi import HTTPException


@pytest.fixture()
def app_module(tmp_path: Path) -> Iterator[object]:
    cfg_src = Path(__file__).resolve().parents[1] / "config" / "train.yaml"
    cfg_dst = tmp_path / "train.yaml"
    cfg_dst.write_text(cfg_src.read_text(encoding="utf-8"), encoding="utf-8")
    os.environ["WEBUI_CONFIG_PATH"] = str(cfg_dst)
    import webui.backend.app as app_module
    importlib.reload(app_module)
    yield app_module


def test_get_config(app_module: object) -> None:
    payload = app_module.get_train_config()
    assert "content" in payload
    assert "path" in payload


def test_update_config_validation(app_module: object) -> None:
    with pytest.raises(HTTPException) as exc:
        app_module.update_train_config({"content": "[]"})
    assert exc.value.status_code == 400

    current = app_module.get_train_config()["content"]
    payload = app_module.update_train_config({"content": current})
    assert payload["ok"] is True


def test_eval_steps_must_be_positive(app_module: object) -> None:
    current = app_module.get_train_config()["content"]
    invalid = "\n".join(
        "eval_steps: 0" if line.startswith("eval_steps:") else line
        for line in current.splitlines()
    )

    with pytest.raises(HTTPException) as exc:
        app_module.update_train_config({"content": invalid})

    assert exc.value.status_code == 400
    assert "eval_steps must be positive" in exc.value.detail
