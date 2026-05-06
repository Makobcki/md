from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, Dict, List, Literal, Optional
import re
import shutil
import time
import uuid
import base64
import hashlib
import hmac
import mimetypes
import secrets
import yaml

from fastapi import Cookie, Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .argparse_reader import parse_argparse_args
from .job_manager import JobManager
from .services import config_service


ROOT_DIR = Path(__file__).resolve().parents[2]
RUNS_DIR = Path(os.environ.get("WEBUI_RUNS_DIR", ROOT_DIR / "webui_runs"))
UPLOADS_DIR = RUNS_DIR / "_uploads"


class WSManager:
    def __init__(self) -> None:
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.connections: Dict[str, Dict[str, List[WebSocket]]] = {}

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    async def connect(self, run_id: str, channel: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.add(run_id, channel, websocket)

    def add(self, run_id: str, channel: str, websocket: WebSocket) -> None:
        self.connections.setdefault(run_id, {}).setdefault(channel, []).append(websocket)

    def disconnect(self, run_id: str, channel: str, websocket: WebSocket) -> None:
        if run_id in self.connections and channel in self.connections[run_id]:
            if websocket in self.connections[run_id][channel]:
                self.connections[run_id][channel].remove(websocket)

    async def broadcast(self, run_id: str, channel: str, message: str) -> None:
        for ws in list(self.connections.get(run_id, {}).get(channel, [])):
            try:
                await ws.send_text(message)
            except RuntimeError:
                self.disconnect(run_id, channel, ws)

    def send_from_thread(self, run_id: str, channel: str, message: str) -> None:
        if not self.loop:
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(run_id, channel, message), self.loop)


logger = logging.getLogger("webui")
_FILE_TOKEN_SECRET = (
    os.environ.get("WEBUI_FILE_TOKEN_SECRET")
    or os.environ.get("WEBUI_AUTH_TOKEN")
    or secrets.token_urlsafe(32)
).encode("utf-8")
_AUTH_COOKIE_NAME = "webui_auth"
_AUTH_COOKIE_MAX_AGE = int(os.environ.get("WEBUI_AUTH_COOKIE_MAX_AGE", str(30 * 24 * 60 * 60)))
_AUTH_COOKIE_SECURE = os.environ.get("WEBUI_AUTH_COOKIE_SECURE", "").strip().lower() in {"1", "true", "yes", "on"}
_AUTH_MAX_FAILED_ATTEMPTS = int(os.environ.get("WEBUI_AUTH_MAX_FAILED_ATTEMPTS", "5"))
_AUTH_LOCKOUT_SECONDS = int(os.environ.get("WEBUI_AUTH_LOCKOUT_SECONDS", "300"))
_AUTH_FAILURE_WINDOW_SECONDS = int(os.environ.get("WEBUI_AUTH_FAILURE_WINDOW_SECONDS", "300"))
_auth_failures: dict[str, dict[str, float | int]] = {}
ws_manager = WSManager()
job_manager = JobManager(ROOT_DIR, ws_manager, RUNS_DIR)


def _cors_allowed_origins() -> list[str]:
    origins = [
        item.strip()
        for item in os.environ.get(
            "WEBUI_CORS_ORIGINS",
            "http://127.0.0.1:5173,http://localhost:5173",
        ).split(",")
        if item.strip()
    ]
    if "*" in origins:
        raise RuntimeError("WEBUI_CORS_ORIGINS='*' is not allowed with cookie authentication.")
    return origins


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    job_manager.set_loop(asyncio.get_running_loop())
    yield


app = FastAPI(lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _get_out_dir() -> Path:
    cfg_path = config_service.get_config_path(ROOT_DIR)
    cfg = config_service.parse_config_text(cfg_path.read_text(encoding="utf-8"))
    out_dir = Path(cfg.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT_DIR / out_dir
    return out_dir


def _normalize_auth_token(value: str | None) -> str:
    if not isinstance(value, str):
        return ""
    token = value.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token


def _token_is_valid(value: str | None) -> bool:
    expected = _normalize_auth_token(os.environ.get("WEBUI_AUTH_TOKEN"))
    if not expected:
        return True
    return hmac.compare_digest(_normalize_auth_token(value), expected)


def _auth_cookie_value(expected: str | None = None) -> str:
    token = _normalize_auth_token(expected if expected is not None else os.environ.get("WEBUI_AUTH_TOKEN"))
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    body = _b64url(
        json.dumps(
            {"v": 1, "digest": digest, "exp": int(time.time()) + _AUTH_COOKIE_MAX_AGE},
            separators=(",", ":"),
        ).encode("utf-8")
    )
    sig = _b64url(hmac.new(_FILE_TOKEN_SECRET, body.encode("ascii"), hashlib.sha256).digest())
    return f"{body}.{sig}"


def _auth_cookie_is_valid(value: str | None) -> bool:
    expected = _normalize_auth_token(os.environ.get("WEBUI_AUTH_TOKEN"))
    if not expected:
        return True
    if not value:
        return False
    try:
        body, sig = str(value).split(".", 1)
        expected_sig = _b64url(hmac.new(_FILE_TOKEN_SECRET, body.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected_sig):
            return False
        payload = json.loads(_b64url_decode(body).decode("utf-8"))
        digest = payload.get("digest") if isinstance(payload, dict) else None
        expires_at = payload.get("exp") if isinstance(payload, dict) else None
        if not isinstance(expires_at, int) or expires_at < int(time.time()):
            return False
        expected_digest = hashlib.sha256(expected.encode("utf-8")).hexdigest()
        return isinstance(digest, str) and hmac.compare_digest(digest, expected_digest)
    except Exception:
        return False


def _set_auth_cookie(response: Response, request: Request | None = None) -> None:
    secure = _AUTH_COOKIE_SECURE or bool(request and request.url.scheme == "https")
    response.set_cookie(
        _AUTH_COOKIE_NAME,
        _auth_cookie_value(),
        max_age=_AUTH_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=secure,
        path="/",
    )


def _clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(_AUTH_COOKIE_NAME, path="/", samesite="lax")


def _auth_client_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
    if forwarded_for:
        return forwarded_for
    real_ip = request.headers.get("x-real-ip", "").strip()
    if real_ip:
        return real_ip
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _auth_retry_after(key: str, now: float | None = None) -> int:
    current = time.time() if now is None else now
    record = _auth_failures.get(key)
    if not record:
        return 0
    locked_until = float(record.get("locked_until", 0.0))
    if locked_until <= current:
        return 0
    return max(1, int(locked_until - current))


def _record_auth_failure(key: str, now: float | None = None) -> int:
    current = time.time() if now is None else now
    record = _auth_failures.get(key)
    if not record or current - float(record.get("first_failed_at", 0.0)) > _AUTH_FAILURE_WINDOW_SECONDS:
        record = {"count": 0, "first_failed_at": current, "locked_until": 0.0}
        _auth_failures[key] = record
    record["count"] = int(record.get("count", 0)) + 1
    if int(record["count"]) >= _AUTH_MAX_FAILED_ATTEMPTS:
        record["locked_until"] = current + _AUTH_LOCKOUT_SECONDS
        return _AUTH_LOCKOUT_SECONDS
    return 0


def _clear_auth_failures(key: str) -> None:
    _auth_failures.pop(key, None)


def _require_token(
    authorization: str | None = Header(default=None),
    auth_cookie: str | None = Cookie(default=None, alias=_AUTH_COOKIE_NAME),
) -> None:
    if _token_is_valid(authorization) or _auth_cookie_is_valid(auth_cookie):
        return
    raise HTTPException(status_code=401, detail="invalid auth token")


async def _require_ws_token(websocket: WebSocket) -> bool:
    expected = _normalize_auth_token(os.environ.get("WEBUI_AUTH_TOKEN"))
    if not expected:
        return True
    authorization = websocket.headers.get("authorization")
    auth_cookie = getattr(websocket, "cookies", {}).get(_AUTH_COOKIE_NAME)
    if _token_is_valid(authorization) or _auth_cookie_is_valid(auth_cookie):
        return True
    await websocket.close(code=1008)
    return False


def _safe_upload_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(name or "upload").name).strip(".-")
    return cleaned or "upload"


def _runs_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(RUNS_DIR.resolve()).as_posix()
    except Exception as exc:
        raise ValueError("path must be inside runs dir") from exc


def _upload_relative_url(path: Path) -> str:
    # Uploaded source images are intentionally not exposed via predictable
    # /api/files/_uploads/... paths. Return a signed preview URL instead.
    return _file_url_for_path(path)


def _bounded_run_artifact_path(value: str | None, *, required: bool = False) -> Path | None:
    if not value:
        if required:
            raise ValueError("path is required")
        return None
    path = Path(value)
    if not path.is_absolute():
        path = RUNS_DIR / path
    resolved = path.resolve()
    allowed_roots = [RUNS_DIR.resolve()]
    try:
        allowed_roots.append(_get_out_dir().resolve())
    except Exception:
        pass
    for item in os.environ.get("WEBUI_ALLOWED_PATHS", "").split(os.pathsep):
        item = item.strip()
        if item:
            allowed_roots.append(Path(item).resolve())
    if not any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots):
        raise ValueError("run artifact path is outside allowed roots")
    return resolved


def _read_bounded_text(value: str, *, limit_lines: int | None = None) -> str:
    path = _bounded_run_artifact_path(value, required=True)
    assert path is not None
    if not path.exists():
        return ""
    if limit_lines is None:
        return path.read_text(encoding="utf-8")
    return _tail_text(path, limit_lines)



def _configured_allowed_roots() -> list[Path]:
    roots = [ROOT_DIR, RUNS_DIR]
    try:
        roots.append(_get_out_dir())
    except Exception as exc:
        logger.warning("failed to resolve configured out_dir for path validation: %s", exc)
    extra = os.environ.get("WEBUI_ALLOWED_PATHS", "")
    for item in extra.split(os.pathsep):
        item = item.strip()
        if item:
            roots.append(Path(item))
    return roots


def _configured_allowed_files() -> list[Path]:
    try:
        return [config_service.get_config_path(ROOT_DIR).resolve()]
    except Exception as exc:
        logger.warning("failed to resolve configured config path for path validation: %s", exc)
        return []


def _bounded_path(value: str | None, *, required: bool = False) -> str | None:
    if not value:
        if required:
            raise ValueError("path is required")
        return value
    path = Path(value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    resolved = path.resolve()
    allowed_roots = [root.resolve() for root in _configured_allowed_roots()]
    allowed_files = _configured_allowed_files()
    if resolved not in allowed_files and not any(
        resolved == root or resolved.is_relative_to(root) for root in allowed_roots
    ):
        allowed = ", ".join(str(root) for root in [*allowed_roots, *allowed_files])
        raise ValueError(f"path must be inside an allowed root: {allowed}")
    return str(resolved)


class SampleArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    ckpt: str
    out: Optional[str] = None
    n: int = Field(default=8, ge=1, le=64)
    steps: int = Field(default=30, ge=1, le=500)
    prompt: str = ""
    neg_prompt: str = ""
    cfg: float = Field(default=5.0, ge=0.0, le=30.0)
    shift: Optional[float] = Field(default=None, gt=0.0, le=100.0)
    sampler: Literal["flow_euler", "flow_heun"] = "flow_heun"
    seed: Optional[int] = 42
    device: Literal["auto", "cpu", "cuda"] = "auto"
    width: Optional[int] = Field(default=None, ge=1)
    height: Optional[int] = Field(default=None, ge=1)
    task: Literal["txt2img", "img2img", "inpaint", "control"] = "txt2img"
    init_image: str = Field(default="", alias="init-image")
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    mask: str = ""
    control_image: str = Field(default="", alias="control-image")
    control_strength: float = Field(default=1.0, ge=0.0, alias="control-strength")
    control_type: Literal["none", "latent_identity", "image", "canny", "depth", "pose", "lineart", "normal"] = Field(default="image", alias="control-type")
    latent_only: bool = Field(default=False, alias="latent-only")
    fake_vae: bool = Field(default=False, alias="fake-vae")
    use_ema: bool = Field(default=True, alias="use-ema")

    @model_validator(mode="after")
    def _validate_paths(self) -> "SampleArgs":
        self.ckpt = _bounded_path(self.ckpt, required=True) or self.ckpt
        self.out = _bounded_path(self.out)
        self.init_image = _bounded_path(self.init_image) or ""
        self.mask = _bounded_path(self.mask) or ""
        self.control_image = _bounded_path(self.control_image) or ""
        supplied = {
            name
            for name, present in {
                "init-image": bool(self.init_image),
                "mask": bool(self.mask),
                "control-image": bool(self.control_image),
            }.items()
            if present
        }
        allowed = {
            "txt2img": set(),
            "img2img": {"init-image"},
            "inpaint": {"init-image", "mask"},
            "control": {"control-image"},
        }[self.task]
        missing = sorted(allowed - supplied)
        extra = sorted(supplied - allowed)
        if missing:
            raise ValueError(f"task={self.task} requires: {', '.join(missing)}")
        if extra:
            raise ValueError(f"task={self.task} does not allow: {', '.join(extra)}")
        out_suffix = Path(self.out or "").suffix.lower()
        if self.latent_only and out_suffix and out_suffix not in {".pt", ".pth"}:
            raise ValueError("latent-only outputs must use .pt or .pth extension")
        if not self.latent_only and out_suffix and out_suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("image outputs must use .png, .jpg, .jpeg or .webp extension")
        return self

    def to_cli_args(self) -> Dict[str, Any]:
        data = self.model_dump(by_alias=True, exclude_none=True)
        if data.get("device") == "auto":
            data.pop("device")
        if data.get("use-ema") is True:
            data.pop("use-ema", None)
        return data

    def to_sample_options_args(self) -> Dict[str, Any]:
        data = self.model_dump(by_alias=False, exclude_none=True)
        if data.get("device") == "auto":
            data.pop("device")
        return data


class SampleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    args: SampleArgs


class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resume: Optional[str] = None

    @model_validator(mode="after")
    def _validate_resume(self) -> "TrainRequest":
        self.resume = _bounded_path(self.resume)
        return self


class LatentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    config: str = "./config/train.yaml"
    overwrite: bool = False
    limit: Optional[int] = Field(default=None, ge=1)
    batch_size: int = Field(default=16, ge=1, le=512, alias="batch-size")
    num_workers: int = Field(default=4, ge=0, le=64, alias="num-workers")
    prefetch_factor: int = Field(default=2, ge=1, le=32, alias="prefetch-factor")
    pin_memory: bool = Field(default=True, alias="pin-memory")
    device: Literal["auto", "cpu", "cuda"] = "auto"
    latent_dtype: Literal["fp16", "bf16"] = Field(default="fp16", alias="latent-dtype")
    autocast_dtype: Literal["fp16", "bf16"] = Field(default="fp16", alias="autocast-dtype")
    queue_size: int = Field(default=64, ge=1, le=4096, alias="queue-size")
    writer_threads: int = Field(default=1, ge=1, le=16, alias="writer-threads")
    shard_size: int = Field(default=4096, ge=0, alias="shard-size")
    stats_every_sec: float = Field(default=5.0, gt=0.0, le=3600.0, alias="stats-every-sec")
    decode_backend: Literal["auto", "pil", "torchvision"] = Field(default="auto", alias="decode-backend")

    @model_validator(mode="after")
    def _validate_config(self) -> "LatentArgs":
        self.config = _bounded_path(self.config, required=True) or self.config
        return self

    def to_cli_args(self) -> Dict[str, Any]:
        data = self.model_dump(by_alias=True, exclude_none=True, exclude_defaults=True)
        data["config"] = self.config
        if data.get("device") == "auto":
            data.pop("device")
        return data


class LatentCacheRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    args: LatentArgs


class AuthLoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str = ""


_LATENT_PREPARE_ARG_KEYS = {
    "overwrite",
    "limit",
    "batch-size",
    "num-workers",
    "prefetch-factor",
    "pin-memory",
    "device",
    "latent-dtype",
    "autocast-dtype",
    "queue-size",
    "writer-threads",
    "shard-size",
    "stats-every-sec",
    "decode-backend",
}


def _latent_prepare_arg_defaults() -> Dict[str, Any]:
    cfg_path = config_service.get_config_path(ROOT_DIR)
    cfg = config_service.parse_config_text(cfg_path.read_text(encoding="utf-8"))
    defaults: Dict[str, Any] = {
        "config": str(cfg_path),
        "latent-dtype": cfg.latent_dtype,
        "autocast-dtype": cfg.latent_dtype,
        "shard-size": 4096 if bool(cfg.latent_cache_sharded) else 0,
    }

    section = cfg.extra.get("prepare_latents", cfg.extra.get("latent_prepare"))
    if isinstance(section, dict):
        for key, value in section.items():
            name = str(key).replace("_", "-")
            if name in _LATENT_PREPARE_ARG_KEYS:
                defaults[name] = value

    prefix = "latent_prepare_"
    for key, value in cfg.extra.items():
        if key.startswith(prefix):
            name = key[len(prefix):].replace("_", "-")
            if name in _LATENT_PREPARE_ARG_KEYS:
                defaults[name] = value
    return defaults


_IMAGE_FILE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_DOWNLOAD_FILE_EXTS = _IMAGE_FILE_EXTS | {".pt", ".pth", ".safetensors", ".json", ".yaml", ".yml", ".txt", ".log"}


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode((value + "=" * (-len(value) % 4)).encode("ascii"))


def _encode_file_token(path: Path) -> str:
    payload = json.dumps({"path": str(path.resolve())}, separators=(",", ":")).encode("utf-8")
    body = _b64url(payload)
    sig = _b64url(hmac.new(_FILE_TOKEN_SECRET, body.encode("ascii"), hashlib.sha256).digest())
    return f"{body}.{sig}"


def _decode_file_token(token: str) -> Path:
    try:
        body, sig = token.split(".", 1)
        expected = _b64url(hmac.new(_FILE_TOKEN_SECRET, body.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            raise ValueError("invalid file token signature")
        payload = json.loads(_b64url_decode(body).decode("utf-8"))
        path = payload.get("path") if isinstance(payload, dict) else None
        if not isinstance(path, str) or not path:
            raise ValueError("invalid file token payload")
        return Path(path)
    except Exception as exc:
        raise ValueError("invalid file token") from exc


def _safe_file_path_from_token(token: str, *, preview_only: bool) -> Path:
    path = _bounded_run_artifact_path(str(_decode_file_token(token)), required=True)
    assert path is not None
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(str(path))
    allowed = _IMAGE_FILE_EXTS if preview_only else _DOWNLOAD_FILE_EXTS
    if path.suffix.lower() not in allowed:
        raise PermissionError("unsupported file type")
    return path


def _file_url_for_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(RUNS_DIR.resolve())
        if rel.parts and rel.parts[0] == "_uploads":
            return f"/api/files/by-path/{_encode_file_token(resolved)}"
        return f"/api/files/{rel.as_posix()}"
    except Exception:
        return f"/api/files/by-path/{_encode_file_token(resolved)}"


def _download_url_for_path(path: Path) -> str:
    return f"/api/files/download/{_encode_file_token(path)}"


def _artifact_record(path: Path, *, source: str, run_id: str | None = None, preview: bool | None = None) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    is_image = suffix in _IMAGE_FILE_EXTS
    previewable = bool(is_image if preview is None else preview)
    stat = path.stat()
    return {
        "path": str(path),
        "name": path.name,
        "source": source,
        "run_id": run_id,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "previewable": previewable,
        "url": _file_url_for_path(path) if previewable else None,
        "download_url": _download_url_for_path(path),
        "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    }


def _iter_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    if not root.exists():
        return []
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.rglob(pattern))
    return [p for p in paths if p.is_file() or p.is_symlink()]


def _collect_webui_samples(run_id: str | None = None, include_latents: bool = False) -> list[Dict[str, Any]]:
    if run_id:
        run_roots = [RUNS_DIR / run_id]
    elif RUNS_DIR.exists():
        run_roots = [p for p in RUNS_DIR.iterdir() if p.is_dir() and p.name != "_uploads"]
    else:
        run_roots = []
    records: list[Dict[str, Any]] = []
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp") + (("*.pt", "*.pth") if include_latents else ())
    for run_root in run_roots:
        samples_root = run_root / "samples"
        for path in _iter_files(samples_root, patterns):
            source = "webui_latent" if path.suffix.lower() in {".pt", ".pth"} else "webui_sample"
            records.append(_artifact_record(path, source=source, run_id=run_root.name, preview=path.suffix.lower() in _IMAGE_FILE_EXTS))
    return records


def _collect_train_samples(run_id: str | None = None) -> list[Dict[str, Any]]:
    roots: list[Path] = []
    if run_id:
        run = job_manager.runs.get(run_id)
        if run is not None:
            notes = run.notes or {}
            for key in ("train_run_dir", "output_dir", "out_dir", "latest_artifact_dir"):
                value = notes.get(key)
                if value:
                    roots.append(Path(str(value)))
            roots.append(Path(run.run_dir))
    else:
        try:
            roots.append(_get_out_dir())
        except Exception:
            pass
    records: list[Dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        candidates = [root / "eval", root / "samples"]
        if root.name in {"eval", "samples"}:
            candidates.append(root)
        for sub in candidates:
            for path in _iter_files(sub, ("*.png", "*.jpg", "*.jpeg", "*.webp")):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                records.append(_artifact_record(path, source="train_sample", run_id=run_id, preview=True))
    return records


def _collect_run_checkpoints(run_id: str | None = None) -> list[Dict[str, Any]]:
    roots: list[Path] = []
    if run_id:
        run = job_manager.runs.get(run_id)
        if run is not None:
            roots.extend([Path(run.run_dir), Path(run.run_dir) / "checkpoints"])
            notes = run.notes or {}
            for key in ("train_run_dir", "output_dir", "out_dir"):
                value = notes.get(key)
                if value:
                    roots.append(Path(str(value)))
    else:
        try:
            roots.append(_get_out_dir())
        except Exception:
            pass
    records: list[Dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        for path in _iter_files(root, ("*.pt", "*.pth", "*.safetensors")):
            if "cache" in {part.lower() for part in path.parts}:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            records.append(_artifact_record(path, source="checkpoint", run_id=run_id, preview=False))
    return records


@app.get("/api/files/by-path/{token}")
def get_safe_file_by_token(token: str, _: None = Depends(_require_token)) -> FileResponse:
    try:
        path = _safe_file_path_from_token(token, preview_only=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid file token") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="file not found") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="only image preview files are exposed") from exc
    return FileResponse(path)


@app.get("/api/files/download/{token}")
def download_safe_file(token: str, _: None = Depends(_require_token)) -> FileResponse:
    try:
        path = _safe_file_path_from_token(token, preview_only=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid file token") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="file not found") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="unsupported file type") from exc
    return FileResponse(path, filename=path.name)


@app.get("/api/auth/status")
def get_auth_status(
    authorization: str | None = Header(default=None),
    auth_cookie: str | None = Cookie(default=None, alias=_AUTH_COOKIE_NAME),
) -> Dict[str, Any]:
    required = bool(_normalize_auth_token(os.environ.get("WEBUI_AUTH_TOKEN")))
    authenticated = (
        (not required)
        or _token_is_valid(authorization)
        or _auth_cookie_is_valid(auth_cookie)
    )
    return {"auth_required": required, "authenticated": authenticated}


@app.post("/api/auth/login")
def login(payload: AuthLoginRequest, response: Response, request: Request) -> Dict[str, Any]:
    required = bool(_normalize_auth_token(os.environ.get("WEBUI_AUTH_TOKEN")))
    client_key = _auth_client_key(request)
    retry_after = _auth_retry_after(client_key)
    if required and retry_after > 0:
        raise HTTPException(
            status_code=429,
            detail=f"too many failed auth attempts; retry after {retry_after} seconds",
            headers={"Retry-After": str(retry_after)},
        )
    if required and not _token_is_valid(payload.token):
        _clear_auth_cookie(response)
        retry_after = _record_auth_failure(client_key)
        headers = {"Retry-After": str(retry_after)} if retry_after > 0 else None
        status_code = 429 if retry_after > 0 else 401
        detail = (
            f"too many failed auth attempts; retry after {retry_after} seconds"
            if retry_after > 0
            else "invalid auth token"
        )
        raise HTTPException(status_code=status_code, detail=detail, headers=headers)
    _clear_auth_failures(client_key)
    if required:
        _set_auth_cookie(response, request)
    else:
        _clear_auth_cookie(response)
    return {"auth_required": required, "authenticated": True}


@app.post("/api/auth/logout")
def logout(response: Response) -> Dict[str, Any]:
    _clear_auth_cookie(response)
    return {"ok": True}


@app.get("/api/files/{rel_path:path}")
def get_safe_file(rel_path: str, _: None = Depends(_require_token)) -> FileResponse:
    path = (RUNS_DIR / rel_path).resolve()
    try:
        rel = path.relative_to(RUNS_DIR.resolve())
    except Exception as exc:
        raise HTTPException(status_code=403, detail="file path is outside runs dir") from exc
    if rel.parts and rel.parts[0] == "_uploads":
        raise HTTPException(status_code=403, detail="uploads require signed file URLs")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    if path.suffix.lower() not in _IMAGE_FILE_EXTS:
        raise HTTPException(status_code=403, detail="only image preview files are exposed")
    return FileResponse(path)


@app.get("/api/status")
def get_status(_: None = Depends(_require_token)) -> Dict[str, Any]:
    return job_manager.get_status()


@app.get("/api/runs")
def list_runs(_: None = Depends(_require_token)) -> List[Dict[str, Any]]:
    return [json.loads(json.dumps(r.__dict__)) for r in job_manager.list_runs()]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, _: None = Depends(_require_token)) -> Dict[str, Any]:
    try:
        run = job_manager.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return json.loads(json.dumps(run.__dict__))


@app.get("/api/runs/{run_id}/config")
def get_run_config(run_id: str, _: None = Depends(_require_token)) -> Dict[str, Any]:
    try:
        run = job_manager.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not run.config_snapshot:
        raise HTTPException(status_code=404, detail="No config snapshot")
    return {"path": run.config_snapshot, "content": _read_bounded_text(run.config_snapshot)}


def _tail_text(path: Path, limit: int) -> str:
    if limit <= 0:
        return ""
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.rstrip("\n"))
            if len(lines) > limit:
                lines.pop(0)
    return "\n".join(lines)


async def _send_log_backlog(websocket: WebSocket, run_id: str, limit: int) -> None:
    if limit <= 0:
        return
    try:
        run = job_manager.get_run(run_id)
    except KeyError:
        return

    for stream, path_value in (("stdout", run.log_stdout), ("stderr", run.log_stderr)):
        path = _bounded_run_artifact_path(path_value)
        if path is None or not path.exists():
            continue
        for line in _tail_text(path, limit).splitlines():
            await websocket.send_text(json.dumps({
                "type": "log",
                "stream": stream,
                "line": JobManager.format_log_line(line),
                "backlog": True,
            }, ensure_ascii=False))


@app.get("/api/runs/{run_id}/logs/{stream}")
def get_run_log(
    run_id: str,
    stream: Literal["stdout", "stderr"],
    limit: Annotated[int, Query(ge=1, le=50000)] = 2000,
    raw: Annotated[bool, Query()] = False,
    _: None = Depends(_require_token),
) -> Dict[str, Any]:
    if stream not in {"stdout", "stderr"}:
        raise HTTPException(status_code=400, detail="invalid log stream")
    try:
        run = job_manager.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    path = _bounded_run_artifact_path(run.log_stdout if stream == "stdout" else run.log_stderr)
    if path is None or not path.exists():
        return {"content": "", "raw": raw}
    lines = _tail_text(path, limit).splitlines()
    content = "\n".join(lines if raw else (JobManager.format_log_line(line) for line in lines))
    return {"content": content, "raw": raw}


@app.get("/api/runs/{run_id}/metrics")
def get_run_metrics(
    run_id: str,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=50000)] = 2000,
    _: None = Depends(_require_token),
) -> Dict[str, Any]:
    try:
        run = job_manager.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not run.metrics_path:
        return {"items": []}
    path = _bounded_run_artifact_path(run.metrics_path)
    if path is None or not path.exists():
        return {"items": []}
    items = []
    next_offset = offset
    for line_no, line in enumerate(path.open("r", encoding="utf-8")):
        if line_no < offset:
            continue
        if len(items) >= limit:
            break
        next_offset = line_no + 1
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"items": items, "offset": offset, "next_offset": next_offset}


@app.get("/api/config")
def get_train_config(_: None = Depends(_require_token)) -> Dict[str, Any]:
    cfg_path = config_service.get_config_path(ROOT_DIR)
    return {"path": str(cfg_path), "content": config_service.read_config_text(ROOT_DIR)}


@app.put("/api/config")
def update_train_config(payload: Dict[str, Any], _: None = Depends(_require_token)) -> Dict[str, Any]:
    content = payload.get("content")
    if not isinstance(content, str):
        raise HTTPException(status_code=400, detail="content must be a string")
    try:
        cfg = config_service.write_config_text(ROOT_DIR, content)
    except (ValueError, yaml.YAMLError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "config": cfg}


@app.get("/api/checkpoints")
def list_checkpoints(_: None = Depends(_require_token)) -> Dict[str, Any]:
    return {"items": job_manager.list_checkpoints()}


@app.get("/api/checkpoints/info")
def get_checkpoint_info(path: str, _: None = Depends(_require_token)) -> Dict[str, Any]:
    if not path:
        raise HTTPException(status_code=400, detail="path is required")
    try:
        ckpt_path = Path(_bounded_path(path, required=True) or path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid checkpoint path") from exc
    if not ckpt_path.exists():
        raise HTTPException(status_code=404, detail="checkpoint not found")

    sidecar = ckpt_path.with_suffix(ckpt_path.suffix + ".metadata.json")
    if sidecar.exists():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail="checkpoint metadata sidecar is invalid") from exc
        cfg = payload.get("cfg", {}) if isinstance(payload, dict) else {}
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        use_text_conditioning = bool(cfg.get("use_text_conditioning", True)) if isinstance(cfg, dict) else True
        return {
            "path": str(ckpt_path),
            "metadata_path": str(sidecar),
            "step": payload.get("step") if isinstance(payload, dict) else None,
            "architecture": metadata.get("architecture") if isinstance(metadata, dict) else payload.get("architecture"),
            "objective": metadata.get("objective") if isinstance(metadata, dict) else payload.get("objective"),
            "prediction_type": metadata.get("prediction_type") if isinstance(metadata, dict) else payload.get("prediction_type"),
            "use_text_conditioning": use_text_conditioning,
        }

    # Legacy fallback for old checkpoints without sidecars. This endpoint is
    # authenticated and path-bounded; new checkpoints avoid this heavy path.
    import torch
    from diffusion.utils import load_ckpt

    ck = load_ckpt(str(ckpt_path), torch.device("cpu"))
    cfg = ck.get("cfg", {})
    use_text_conditioning = bool(cfg.get("use_text_conditioning", True))
    meta_flag = ck.get("meta", {}).get("use_text_conditioning")
    if isinstance(meta_flag, bool):
        use_text_conditioning = meta_flag
    metadata = ck.get("metadata", {}) if isinstance(ck.get("metadata", {}), dict) else {}
    return {
        "path": str(ckpt_path),
        "step": ck.get("step", metadata.get("step")),
        "architecture": metadata.get("architecture", ck.get("architecture")),
        "objective": metadata.get("objective", ck.get("objective")),
        "prediction_type": metadata.get("prediction_type", ck.get("prediction_type")),
        "use_text_conditioning": use_text_conditioning,
    }


@app.get("/api/out_dir/summary")
def get_out_dir_summary(_: None = Depends(_require_token)) -> Dict[str, Any]:
    out_dir = _get_out_dir()
    train_log = out_dir / "metrics" / "events.jsonl"
    config_snapshot = out_dir / "config_snapshot.yaml"
    run_meta = out_dir / "run_meta.yaml"

    def _tail(path: Path, limit: int = 200) -> str:
        if not path.exists():
            return ""
        lines = path.read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[-limit:])

    return {
        "out_dir": str(out_dir),
        "train_log": {"path": str(train_log), "tail": _tail(train_log)},
        "config_snapshot": {
            "path": str(config_snapshot),
            "content": config_snapshot.read_text(encoding="utf-8") if config_snapshot.exists() else "",
        },
        "run_meta": {
            "path": str(run_meta),
            "content": run_meta.read_text(encoding="utf-8") if run_meta.exists() else "",
        },
        "checkpoints": sorted([str(p) for p in out_dir.rglob("*.pt")]) if out_dir.exists() else [],
    }


def _sample_arg_specs() -> list[Dict[str, Any]]:
    return [
        {"name": "ckpt", "flags": ["--ckpt"], "type": "str", "default": None, "required": True, "help": "Checkpoint path", "choices": None},
        {"name": "out", "flags": ["--out"], "type": "str", "default": "", "required": False, "help": "Optional output path. Defaults to this WebUI run samples directory.", "choices": None},
        {"name": "task", "flags": ["--task"], "type": "str", "default": "txt2img", "required": False, "help": "Generation task", "choices": ["txt2img", "img2img", "inpaint", "control"]},
        {"name": "sampler", "flags": ["--sampler"], "type": "str", "default": "flow_heun", "required": False, "help": "Flow sampler", "choices": ["flow_euler", "flow_heun"]},
        {"name": "n", "flags": ["--n"], "type": "int", "default": 1, "required": False, "help": "Number of images", "choices": None},
        {"name": "steps", "flags": ["--steps"], "type": "int", "default": 30, "required": False, "help": "Sampling steps", "choices": None},
        {"name": "cfg", "flags": ["--cfg"], "type": "float", "default": 5.0, "required": False, "help": "Classifier-free guidance scale", "choices": None},
        {"name": "shift", "flags": ["--shift"], "type": "float", "default": None, "required": False, "help": "Positive inference timestep shift override. Empty = checkpoint/config default.", "choices": None},
        {"name": "seed", "flags": ["--seed"], "type": "int", "default": 42, "required": False, "help": "Base seed. Empty/null lets backend choose a random seed.", "choices": None},
        {"name": "device", "flags": ["--device"], "type": "str", "default": "auto", "required": False, "help": "Runtime device", "choices": ["auto", "cuda", "cpu"]},
        {"name": "width", "flags": ["--width"], "type": "int", "default": None, "required": False, "help": "Output width. Empty = checkpoint/config image_size.", "choices": None},
        {"name": "height", "flags": ["--height"], "type": "int", "default": None, "required": False, "help": "Output height. Empty = checkpoint/config image_size.", "choices": None},
        {"name": "prompt", "flags": ["--prompt"], "type": "str", "default": "", "required": False, "help": "Positive prompt", "choices": None},
        {"name": "neg_prompt", "flags": ["--neg_prompt", "--negative-prompt"], "type": "str", "default": "", "required": False, "help": "Negative prompt. Empty uses cached empty prompt when available.", "choices": None},
        {"name": "init-image", "flags": ["--init-image"], "type": "str", "default": "", "required": False, "help": "Input image for img2img/inpaint", "choices": None},
        {"name": "strength", "flags": ["--strength"], "type": "float", "default": 1.0, "required": False, "help": "Img2img start strength in [0, 1]", "choices": None},
        {"name": "mask", "flags": ["--mask"], "type": "str", "default": "", "required": False, "help": "Inpaint mask path", "choices": None},
        {"name": "control-image", "flags": ["--control-image"], "type": "str", "default": "", "required": False, "help": "Control conditioning image path", "choices": None},
        {"name": "control-strength", "flags": ["--control-strength"], "type": "float", "default": 1.0, "required": False, "help": "Control latent multiplier", "choices": None},
        {"name": "control-type", "flags": ["--control-type"], "type": "str", "default": "image", "required": False, "help": "Control preprocessor type", "choices": ["none", "latent_identity", "image", "canny", "depth", "pose", "lineart", "normal"]},
        {"name": "latent-only", "flags": ["--latent-only"], "type": "bool", "default": False, "required": False, "help": "Write latent tensor instead of image", "choices": None},
        {"name": "fake-vae", "flags": ["--fake-vae"], "type": "bool", "default": False, "required": False, "help": "Use deterministic fake VAE for smoke/sample tests", "choices": None},
        {"name": "use-ema", "flags": ["--use-ema", "--no-ema"], "type": "bool", "default": True, "required": False, "help": "Use EMA weights from checkpoint when available", "choices": None},
    ]


@app.get("/api/sample/args")
def get_sample_args(_: None = Depends(_require_token)) -> Dict[str, Any]:
    return {"items": _sample_arg_specs()}


@app.get("/api/latents/args")
def get_latent_args(_: None = Depends(_require_token)) -> Dict[str, Any]:
    prep_path = ROOT_DIR / "scripts" / "prepare_latents.py"
    items = parse_argparse_args(prep_path)
    defaults = _latent_prepare_arg_defaults()
    for item in items:
        name = item.get("name")
        if name in defaults:
            item["default"] = defaults[name]
    return {"items": items}


@app.get("/api/artifacts")
def list_artifacts(
    run_id: str | None = None,
    source: Literal["all", "webui_samples", "train_samples", "checkpoints", "latents"] = "all",
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
    _: None = Depends(_require_token),
) -> Dict[str, Any]:
    records: list[Dict[str, Any]] = []
    if source in {"all", "webui_samples", "latents"}:
        records.extend(_collect_webui_samples(run_id, include_latents=source in {"all", "latents"}))
    if source in {"all", "train_samples"}:
        records.extend(_collect_train_samples(run_id))
    if source in {"all", "checkpoints"}:
        records.extend(_collect_run_checkpoints(run_id))
    records = sorted(records, key=lambda item: float(item.get("mtime") or 0.0))
    return {"items": records[offset:offset + limit], "offset": offset, "total": len(records)}


@app.get("/api/generated-samples")
def list_generated_samples(
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
    run_id: str | None = None,
    _: None = Depends(_require_token),
) -> Dict[str, Any]:
    records = sorted(_collect_webui_samples(run_id, include_latents=False), key=lambda item: float(item.get("mtime") or 0.0))
    return {"items": records[offset:offset + limit], "offset": offset, "total": len(records)}


@app.get("/api/train-samples")
def list_train_samples(
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
    run_id: str | None = None,
    _: None = Depends(_require_token),
) -> Dict[str, Any]:
    records = sorted(_collect_train_samples(run_id), key=lambda item: float(item.get("mtime") or 0.0))
    return {"items": records[offset:offset + limit], "offset": offset, "total": len(records)}


@app.get("/api/samples")
def list_samples(
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
    run_id: str | None = None,
    _: None = Depends(_require_token),
) -> Dict[str, Any]:
    # Backward-compatible alias for generated WebUI sample images only.
    records = sorted(_collect_webui_samples(run_id, include_latents=False), key=lambda item: float(item.get("mtime") or 0.0))
    return {"items": records[offset:offset + limit], "offset": offset, "total": len(records)}


@app.post("/api/uploads/image")
async def upload_image(
    file: UploadFile = File(...),
    kind: str = Form(default="image"),
    _: None = Depends(_require_token),
) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="file is required")
    content_type = (file.content_type or "").lower()
    ext = Path(file.filename).suffix.lower()
    allowed_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    allowed_types = {"image/png", "image/jpeg", "image/webp", "image/bmp"}
    if ext not in allowed_exts or content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="unsupported image format")
    safe_kind = re.sub(r"[^A-Za-z0-9_-]+", "-", kind or "image").strip("-") or "image"
    safe_name = _safe_upload_name(file.filename)
    out_dir = UPLOADS_DIR / safe_kind
    out_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(float(os.environ.get("WEBUI_MAX_UPLOAD_MB", "32")) * 1024 * 1024)
    tmp_path = out_dir / f".{uuid.uuid4().hex}.upload"
    written = 0
    try:
        with tmp_path.open("wb") as f:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(status_code=413, detail="uploaded image is too large")
                f.write(chunk)
        try:
            from PIL import Image
            with Image.open(tmp_path) as im:
                fmt = (im.format or "").upper()
                im.verify()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid image file") from exc
        suffix_map = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "BMP": ".bmp"}
        suffix = suffix_map.get(fmt)
        if suffix is None:
            raise HTTPException(status_code=400, detail="unsupported image format")
        out_path = out_dir / f"{uuid.uuid4().hex[:12]}_{Path(safe_name).stem}{suffix}"
        tmp_path.replace(out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
        await file.close()
    return {
        "ok": True,
        "path": str(out_path.resolve()),
        "url": _upload_relative_url(out_path),
        "download_url": _download_url_for_path(out_path),
        "name": safe_name,
        "kind": safe_kind,
        "content_type": file.content_type or "",
        "size": out_path.stat().st_size,
    }


@app.post("/api/train/start")
def start_train(req: TrainRequest | None = None, _: None = Depends(_require_token)) -> Dict[str, Any]:
    try:
        resume = req.resume if req else None
        run = job_manager.start_train(resume=resume)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"run_id": run.run_id, "command": run.command}


@app.post("/api/train/stop")
def stop_train(_: None = Depends(_require_token)) -> Dict[str, Any]:
    try:
        run = job_manager.stop_current(expected_type="train")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not run:
        raise HTTPException(status_code=404, detail="No running job")
    return {"run_id": run.run_id}


@app.post("/api/sample/start")
def start_sample(req: SampleRequest, _: None = Depends(_require_token)) -> Dict[str, Any]:
    try:
        run = job_manager.start_sample(req.args.to_sample_options_args())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"run_id": run.run_id, "command": run.command, "output": run.output_path}


@app.post("/api/latents/start")
def start_latent_cache(req: LatentCacheRequest, _: None = Depends(_require_token)) -> Dict[str, Any]:
    try:
        run = job_manager.start_prepare_latents(req.args.to_cli_args())
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"run_id": run.run_id, "command": run.command}


@app.post("/api/sample/stop")
def stop_sample(_: None = Depends(_require_token)) -> Dict[str, Any]:
    try:
        run = job_manager.stop_current(expected_type="sample")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not run:
        raise HTTPException(status_code=404, detail="No running job")
    return {"run_id": run.run_id}


@app.post("/api/latents/stop")
def stop_latent_cache(_: None = Depends(_require_token)) -> Dict[str, Any]:
    try:
        run = job_manager.stop_current(expected_type="latent_cache")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not run:
        raise HTTPException(status_code=404, detail="No running job")
    return {"run_id": run.run_id}


@app.websocket("/ws/logs/{run_id}")
async def ws_logs(
    websocket: WebSocket,
    run_id: str,
    backlog: Annotated[int, Query(ge=0, le=50000)] = 2000,
) -> None:
    if not await _require_ws_token(websocket):
        return
    await websocket.accept()
    try:
        await _send_log_backlog(websocket, run_id, backlog)
        ws_manager.add(run_id, "logs", websocket)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(run_id, "logs", websocket)


@app.websocket("/ws/metrics/{run_id}")
async def ws_metrics(websocket: WebSocket, run_id: str) -> None:
    if not await _require_ws_token(websocket):
        return
    await ws_manager.connect(run_id, "metrics", websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(run_id, "metrics", websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("webui.backend.app:app", host="127.0.0.1", port=8000)


_FRONTEND_DIST = ROOT_DIR / "webui" / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
