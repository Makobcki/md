from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Iterator

import pytest


@pytest.fixture()
def app_module(tmp_path: Path) -> Iterator[object]:
    cfg_src = Path(__file__).resolve().parents[1] / "config" / "train.yaml"
    cfg_dst = tmp_path / "train.yaml"
    cfg_dst.write_text(cfg_src.read_text(encoding="utf-8"), encoding="utf-8")
    os.environ["WEBUI_CONFIG_PATH"] = str(cfg_dst)
    import webui.backend.app as app_module
    importlib.reload(app_module)
    yield app_module


def test_status_endpoint(app_module: object) -> None:
    payload = app_module.get_status()
    assert "active" in payload


def test_latent_args_endpoint(app_module: object) -> None:
    payload = app_module.get_latent_args()
    assert "items" in payload
