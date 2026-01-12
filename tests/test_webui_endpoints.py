from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    cfg_src = Path(__file__).resolve().parents[1] / "config" / "train.yaml"
    cfg_dst = tmp_path / "train.yaml"
    cfg_dst.write_text(cfg_src.read_text(encoding="utf-8"), encoding="utf-8")
    os.environ["WEBUI_CONFIG_PATH"] = str(cfg_dst)
    import webui.backend.app as app_module
    importlib.reload(app_module)
    with TestClient(app_module.app) as client:
        yield client


def test_status_endpoint(client: TestClient) -> None:
    resp = client.get("/api/status")
    assert resp.status_code == 200
    payload = resp.json()
    assert "active" in payload


def test_latent_args_endpoint(client: TestClient) -> None:
    resp = client.get("/api/latents/args")
    assert resp.status_code == 200
    payload = resp.json()
    assert "items" in payload
