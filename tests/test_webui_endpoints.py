from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from typing import Iterator

import pytest
from fastapi import HTTPException, Request, Response

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

    run_dir = app_module.RUNS_DIR / "run-1"
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

    run_dir = app_module.RUNS_DIR / "run-1"
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


def _request(scheme: str = "http") -> Request:
    return Request(
        {
            "type": "http",
            "scheme": scheme,
            "server": ("testserver", 80),
            "client": ("127.0.0.1", 12345),
            "path": "/",
            "headers": [],
        }
    )


def test_websocket_requires_token(app_module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeWebSocket:
        def __init__(
            self,
            authorization: str | None = None,
            auth_cookie: str | None = None,
        ) -> None:
            self.query_params = {}
            self.headers = {"authorization": authorization} if authorization is not None else {}
            self.cookies = {"webui_auth": auth_cookie} if auth_cookie is not None else {}
            self.closed_code = None

        async def close(self, code: int) -> None:
            self.closed_code = code

    monkeypatch.setenv("WEBUI_AUTH_TOKEN", "secret")

    rejected = _FakeWebSocket()
    assert asyncio.run(app_module._require_ws_token(rejected)) is False
    assert rejected.closed_code == 1008

    accepted = _FakeWebSocket(authorization="Bearer secret")
    assert asyncio.run(app_module._require_ws_token(accepted)) is True
    assert accepted.closed_code is None

    cookie = app_module._auth_cookie_value()
    accepted_cookie = _FakeWebSocket(auth_cookie=cookie)
    assert asyncio.run(app_module._require_ws_token(accepted_cookie)) is True
    assert accepted_cookie.closed_code is None


def test_http_auth_accepts_header_or_cookie(app_module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBUI_AUTH_TOKEN", "secret")
    cookie = app_module._auth_cookie_value()

    app_module._require_token(authorization="Bearer secret", auth_cookie=None)
    app_module._require_token(authorization=None, auth_cookie=cookie)

    with pytest.raises(HTTPException) as exc:
        app_module._require_token(authorization=None, auth_cookie="wrong")

    assert exc.value.status_code == 401


def test_auth_token_normalizes_bearer_prefix(app_module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBUI_AUTH_TOKEN", "Bearer secret")

    assert app_module._token_is_valid("secret")
    assert app_module._token_is_valid("Bearer secret")
    assert app_module.get_auth_status(authorization="Bearer secret")["authenticated"] is True


def test_auth_login_sets_cookie_session(app_module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBUI_AUTH_TOKEN", "secret")
    response = Response()

    payload = app_module.login(app_module.AuthLoginRequest(token="secret"), response, _request())
    cookie = response.headers["set-cookie"].split("webui_auth=", 1)[1].split(";", 1)[0]

    assert payload["authenticated"] is True
    assert app_module._auth_cookie_is_valid(cookie)
    app_module._require_token(authorization=None, auth_cookie=cookie)
    assert app_module.get_auth_status(auth_cookie=cookie)["authenticated"] is True


def test_auth_cookie_has_server_side_expiry(app_module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBUI_AUTH_TOKEN", "secret")
    monkeypatch.setattr(app_module, "_AUTH_COOKIE_MAX_AGE", -1)

    assert app_module._auth_cookie_is_valid(app_module._auth_cookie_value()) is False


def test_auth_login_rejects_bad_token(app_module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBUI_AUTH_TOKEN", "secret")

    with pytest.raises(HTTPException) as exc:
        app_module.login(app_module.AuthLoginRequest(token="wrong"), Response(), _request())

    assert exc.value.status_code == 401


def test_auth_login_rate_limits_failed_attempts(app_module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBUI_AUTH_TOKEN", "secret")
    monkeypatch.setattr(app_module, "_AUTH_MAX_FAILED_ATTEMPTS", 2)
    monkeypatch.setattr(app_module, "_AUTH_LOCKOUT_SECONDS", 60)
    app_module._auth_failures.clear()

    with pytest.raises(HTTPException) as first:
        app_module.login(app_module.AuthLoginRequest(token="wrong-1"), Response(), _request())
    with pytest.raises(HTTPException) as second:
        app_module.login(app_module.AuthLoginRequest(token="wrong-2"), Response(), _request())
    with pytest.raises(HTTPException) as locked:
        app_module.login(app_module.AuthLoginRequest(token="secret"), Response(), _request())

    assert first.value.status_code == 401
    assert second.value.status_code == 429
    assert locked.value.status_code == 429
    assert locked.value.headers["Retry-After"]


def test_auth_login_success_clears_failed_attempts(app_module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBUI_AUTH_TOKEN", "secret")
    app_module._auth_failures["127.0.0.1"] = {"count": 1, "first_failed_at": 1.0, "locked_until": 0.0}

    response = Response()
    payload = app_module.login(app_module.AuthLoginRequest(token="secret"), response, _request())

    assert payload["authenticated"] is True
    assert "127.0.0.1" not in app_module._auth_failures


def test_wildcard_cors_is_rejected(app_module: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEBUI_CORS_ORIGINS", "*")

    with pytest.raises(RuntimeError, match="not allowed"):
        app_module._cors_allowed_origins()


def test_artifacts_endpoint_returns_backend_urls(app_module: object) -> None:
    sample_dir = app_module.RUNS_DIR / "run-1" / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    img = sample_dir / "sample.png"
    img.write_bytes(b"png")

    payload = app_module.list_artifacts(run_id="run-1", source="all")

    assert payload["total"] == 1
    item = payload["items"][0]
    assert item["url"].startswith("/api/files/")
    assert item["download_url"].startswith("/api/files/download/")
    assert item["source"] == "webui_sample"


def test_download_token_rejects_unsupported_file(app_module: object) -> None:
    path = app_module.RUNS_DIR / "run-1" / "state.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    token = app_module._encode_file_token(path)

    with pytest.raises(HTTPException) as exc:
        app_module.download_safe_file(token)

    assert exc.value.status_code == 403


def test_file_token_is_signed(app_module: object) -> None:
    path = app_module.RUNS_DIR / "run-1" / "samples" / "sample.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png")
    token = app_module._encode_file_token(path)
    body, sig = token.split(".", 1)
    forged = f"{body}.{'A' * len(sig)}"

    with pytest.raises(HTTPException) as exc:
        app_module.get_safe_file_by_token(forged)

    assert exc.value.status_code == 400


def test_direct_uploads_path_is_forbidden(app_module: object) -> None:
    path = app_module.UPLOADS_DIR / "init" / "source.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png")

    with pytest.raises(HTTPException) as exc:
        app_module.get_safe_file("_uploads/init/source.png")

    assert exc.value.status_code == 403


def test_upload_file_url_uses_signed_token(app_module: object) -> None:
    path = app_module.UPLOADS_DIR / "init" / "source.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png")

    assert app_module._file_url_for_path(path).startswith("/api/files/by-path/")
