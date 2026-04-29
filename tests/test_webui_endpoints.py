from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from typing import Iterator

import pytest
from fastapi import HTTPException


@pytest.fixture()
def app_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    cfg_src = Path(__file__).resolve().parents[1] / "config" / "train.yaml"
    cfg_dst = tmp_path / "train.yaml"
    out_dir = tmp_path / "external_out"
    content = "\n".join(
        f"out_dir: {out_dir}" if line.startswith("out_dir:") else line
        for line in cfg_src.read_text(encoding="utf-8").splitlines()
    )
    cfg_dst.write_text(content, encoding="utf-8")
    monkeypatch.setenv("WEBUI_CONFIG_PATH", str(cfg_dst))
    monkeypatch.setenv("WEBUI_RUNS_DIR", str(tmp_path / "webui_runs"))
    import webui.backend.app as app_module
    importlib.reload(app_module)
    yield app_module


def test_status_endpoint(app_module: object) -> None:
    payload = app_module.get_status()
    assert "active" in payload


def test_latent_args_endpoint(app_module: object) -> None:
    payload = app_module.get_latent_args()
    assert "items" in payload
    device = next(item for item in payload["items"] if item["name"] == "device")
    assert device["default"] == "auto"


def test_import_does_not_create_runs_dir(app_module: object) -> None:
    assert not app_module.RUNS_DIR.exists()


def test_lifespan_creates_runs_dir(app_module: object) -> None:
    async def _run_lifespan() -> None:
        async with app_module._lifespan(app_module.app):
            pass

    asyncio.run(_run_lifespan())

    assert app_module.RUNS_DIR.exists()


def test_sample_checkpoint_allows_configured_out_dir(app_module: object, tmp_path: Path) -> None:
    out_dir = tmp_path / "external_out"
    ckpt = out_dir / "ckpt_0000001.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"")

    args = app_module.SampleArgs(ckpt=str(ckpt))

    assert args.ckpt == str(ckpt.resolve())


def test_latent_args_preserves_auto_device(app_module: object) -> None:
    args = app_module.LatentArgs(config=app_module.get_train_config()["path"], device="auto")

    assert args.device == "auto"


def test_routes_have_unique_path_method_pairs(app_module: object) -> None:
    seen = set()
    duplicates = []
    for route in app_module.app.routes:
        for method in getattr(route, "methods", []) or []:
            key = (method, getattr(route, "path", ""))
            if key in seen:
                duplicates.append(key)
            seen.add(key)

    assert duplicates == []


def test_invalid_log_stream_is_rejected(app_module: object) -> None:
    with pytest.raises(HTTPException) as exc:
        app_module.get_run_log("missing", "stdouterr")

    assert exc.value.status_code == 400


def test_websocket_requires_token(app_module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeWebSocket:
        def __init__(self, token: str | None = None, authorization: str | None = None) -> None:
            self.query_params = {"token": token} if token is not None else {}
            self.headers = {"authorization": authorization} if authorization is not None else {}
            self.closed_code = None

        async def close(self, code: int) -> None:
            self.closed_code = code

    monkeypatch.setenv("WEBUI_AUTH_TOKEN", "secret")

    rejected = _FakeWebSocket()
    assert asyncio.run(app_module._require_ws_token(rejected)) is False
    assert rejected.closed_code == 1008

    accepted = _FakeWebSocket(token="secret")
    assert asyncio.run(app_module._require_ws_token(accepted)) is True
    assert accepted.closed_code is None
