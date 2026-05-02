from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from typing import Iterator

import pytest
from fastapi import HTTPException

from webui.backend.job_manager import RunRecord


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


def test_latent_args_endpoint_reads_prepare_config_defaults(app_module: object) -> None:
    cfg_path = Path(app_module.get_train_config()["path"])
    content = cfg_path.read_text(encoding="utf-8")
    if "latent_prepare_batch_size:" in content:
        content = "\n".join(
            "latent_prepare_batch_size: 23" if line.startswith("latent_prepare_batch_size:") else line
            for line in content.splitlines()
        )
    else:
        content += "\nlatent_prepare_batch_size: 23\n"
    cfg_path.write_text(content, encoding="utf-8")

    payload = app_module.get_latent_args()
    batch_size = next(item for item in payload["items"] if item["name"] == "batch-size")

    assert batch_size["default"] == 23


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


def test_log_websocket_backlog_sends_existing_lines(app_module: object, tmp_path: Path) -> None:
    class _FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send_text(self, message: str) -> None:
            self.sent.append(message)

    run_dir = tmp_path / "run"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True)
    stdout_path = logs_dir / "train.log"
    stderr_path = logs_dir / "train.err.log"
    stdout_path.write_text("out-1\nout-2\nout-3\n", encoding="utf-8")
    stderr_path.write_text("err-1\nerr-2\nerr-3\n", encoding="utf-8")

    app_module.job_manager.runs["run-1"] = RunRecord(
        run_id="run-1",
        run_type="train",
        status="running",
        command=[],
        created_at="",
        pid=None,
        started_at="",
        ended_at=None,
        exit_code=None,
        run_dir=str(run_dir),
        config_snapshot=None,
        log_stdout=str(stdout_path),
        log_stderr=str(stderr_path),
        metrics_path=None,
        output_path=None,
        notes={},
    )

    websocket = _FakeWebSocket()
    asyncio.run(app_module._send_log_backlog(websocket, "run-1", 2))

    payloads = [json.loads(item) for item in websocket.sent]
    assert payloads == [
        {"type": "log", "stream": "stdout", "line": "out-2", "backlog": True},
        {"type": "log", "stream": "stdout", "line": "out-3", "backlog": True},
        {"type": "log", "stream": "stderr", "line": "err-2", "backlog": True},
        {"type": "log", "stream": "stderr", "line": "err-3", "backlog": True},
    ]


def test_log_backlog_formats_json_events(app_module: object, tmp_path: Path) -> None:
    class _FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send_text(self, message: str) -> None:
            self.sent.append(message)

    run_dir = tmp_path / "run"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True)
    stdout_path = logs_dir / "train.log"
    stderr_path = logs_dir / "train.err.log"
    stdout_path.write_text(
        '{"type":"log","message":"saved best checkpoint runs/anime/ckpt_best.pt","step":10,"best_loss":0.42}\n',
        encoding="utf-8",
    )
    stderr_path.write_text("", encoding="utf-8")

    app_module.job_manager.runs["run-1"] = RunRecord(
        run_id="run-1",
        run_type="train",
        status="running",
        command=[],
        created_at="",
        pid=None,
        started_at="",
        ended_at=None,
        exit_code=None,
        run_dir=str(run_dir),
        config_snapshot=None,
        log_stdout=str(stdout_path),
        log_stderr=str(stderr_path),
        metrics_path=None,
        output_path=None,
        notes={},
    )

    websocket = _FakeWebSocket()
    asyncio.run(app_module._send_log_backlog(websocket, "run-1", 2))

    payloads = [json.loads(item) for item in websocket.sent]
    assert payloads == [
        {
            "type": "log",
            "stream": "stdout",
            "line": "step=10 saved best checkpoint runs/anime/ckpt_best.pt best_loss=0.42",
            "backlog": True,
        },
    ]


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
