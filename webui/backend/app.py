from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Literal, Optional
import re
import shutil
import uuid

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
ws_manager = WSManager()
job_manager = JobManager(ROOT_DIR, ws_manager, RUNS_DIR)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    job_manager.set_loop(asyncio.get_running_loop())
    yield


app = FastAPI(lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        item.strip()
        for item in os.environ.get(
            "WEBUI_CORS_ORIGINS",
            "http://127.0.0.1:5173,http://localhost:5173",
        ).split(",")
        if item.strip()
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/runs", StaticFiles(directory=str(RUNS_DIR), check_dir=False), name="runs")


def _get_out_dir() -> Path:
    cfg_path = config_service.get_config_path(ROOT_DIR)
    cfg = config_service.parse_config_text(cfg_path.read_text(encoding="utf-8"))
    out_dir = Path(cfg.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT_DIR / out_dir
    return out_dir


def _token_is_valid(value: str | None) -> bool:
    expected = os.environ.get("WEBUI_AUTH_TOKEN")
    if not expected:
        return True
    return value == f"Bearer {expected}" or value == expected


def _require_token(authorization: str | None = Header(default=None)) -> None:
    if _token_is_valid(authorization):
        return
    raise HTTPException(status_code=401, detail="invalid auth token")


async def _require_ws_token(websocket: WebSocket) -> bool:
    expected = os.environ.get("WEBUI_AUTH_TOKEN")
    if not expected:
        return True
    token = websocket.query_params.get("token")
    authorization = websocket.headers.get("authorization")
    if _token_is_valid(authorization) or _token_is_valid(token):
        return True
    await websocket.close(code=1008)
    return False


def _safe_upload_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(name or "upload").name).strip(".-")
    return cleaned or "upload"


def _upload_relative_url(path: Path) -> str:
    try:
        rel = path.resolve().relative_to(RUNS_DIR.resolve())
    except Exception as exc:
        raise ValueError("upload path must be inside runs dir") from exc
    return f"/runs/{rel.as_posix()}"


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
    device: Literal["auto", "cpu", "cuda"] = "cuda"
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
        if self.task in {"img2img", "inpaint"} and not self.init_image:
            raise ValueError(f"task={self.task} requires init-image")
        if self.task == "inpaint" and not self.mask:
            raise ValueError("task=inpaint requires mask")
        if self.task == "control" and not self.control_image:
            raise ValueError("task=control requires control-image")
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


@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    return job_manager.get_status()


@app.get("/api/runs")
def list_runs() -> List[Dict[str, Any]]:
    return [json.loads(json.dumps(r.__dict__)) for r in job_manager.list_runs()]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    try:
        run = job_manager.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return json.loads(json.dumps(run.__dict__))


@app.get("/api/runs/{run_id}/config")
def get_run_config(run_id: str) -> Dict[str, Any]:
    try:
        run = job_manager.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not run.config_snapshot:
        raise HTTPException(status_code=404, detail="No config snapshot")
    return {"path": run.config_snapshot, "content": Path(run.config_snapshot).read_text(encoding="utf-8")}


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
        path = Path(path_value)
        if not path.exists():
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
    limit: int = Query(default=2000, ge=1, le=50000),
) -> Dict[str, Any]:
    if stream not in {"stdout", "stderr"}:
        raise HTTPException(status_code=400, detail="invalid log stream")
    try:
        run = job_manager.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    path = Path(run.log_stdout if stream == "stdout" else run.log_stderr)
    if not path.exists():
        return {"content": ""}
    content = "\n".join(JobManager.format_log_line(line) for line in _tail_text(path, limit).splitlines())
    return {"content": content}


@app.get("/api/runs/{run_id}/metrics")
def get_run_metrics(
    run_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=2000, ge=1, le=50000),
) -> Dict[str, Any]:
    try:
        run = job_manager.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not run.metrics_path:
        return {"items": []}
    path = Path(run.metrics_path)
    if not path.exists():
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
def get_train_config() -> Dict[str, Any]:
    cfg_path = config_service.get_config_path(ROOT_DIR)
    return {"path": str(cfg_path), "content": config_service.read_config_text(ROOT_DIR)}


@app.put("/api/config")
def update_train_config(payload: Dict[str, Any], _: None = Depends(_require_token)) -> Dict[str, Any]:
    content = payload.get("content")
    if not isinstance(content, str):
        raise HTTPException(status_code=400, detail="content must be a string")
    try:
        cfg = config_service.write_config_text(ROOT_DIR, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "config": cfg}


@app.get("/api/checkpoints")
def list_checkpoints() -> Dict[str, Any]:
    return {"items": job_manager.list_checkpoints()}


@app.get("/api/checkpoints/info")
def get_checkpoint_info(path: str) -> Dict[str, Any]:
    if not path:
        raise HTTPException(status_code=400, detail="path is required")
    try:
        ckpt_path = Path(_bounded_path(path, required=True) or path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid checkpoint path") from exc
    if not ckpt_path.exists():
        raise HTTPException(status_code=404, detail="checkpoint not found")
    import torch
    from diffusion.utils import load_ckpt

    ck = load_ckpt(str(ckpt_path), torch.device("cpu"))
    cfg = ck.get("cfg", {})
    use_text_conditioning = bool(cfg.get("use_text_conditioning", True))
    meta_flag = ck.get("meta", {}).get("use_text_conditioning")
    if isinstance(meta_flag, bool):
        use_text_conditioning = meta_flag
    return {"path": str(ckpt_path), "use_text_conditioning": use_text_conditioning}


@app.get("/api/out_dir/summary")
def get_out_dir_summary() -> Dict[str, Any]:
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
        {"name": "prompt", "flags": ["--prompt"], "type": "str", "default": "", "required": False, "help": "Positive prompt", "choices": None},
        {"name": "neg_prompt", "flags": ["--neg_prompt", "--negative-prompt"], "type": "str", "default": "", "required": False, "help": "Negative prompt. Empty uses cached empty prompt when available.", "choices": None},
        {"name": "init-image", "flags": ["--init-image"], "type": "str", "default": "", "required": False, "help": "Input image for img2img/inpaint", "choices": None},
        {"name": "strength", "flags": ["--strength"], "type": "float", "default": 1.0, "required": False, "help": "Img2img start strength in [0, 1]", "choices": None},
        {"name": "mask", "flags": ["--mask"], "type": "str", "default": "", "required": False, "help": "Inpaint mask path", "choices": None},
        {"name": "control-image", "flags": ["--control-image"], "type": "str", "default": "", "required": False, "help": "Control conditioning image path", "choices": None},
        {"name": "control-strength", "flags": ["--control-strength"], "type": "float", "default": 1.0, "required": False, "help": "Control latent multiplier", "choices": None},
        {"name": "latent-only", "flags": ["--latent-only"], "type": "bool", "default": False, "required": False, "help": "Write latent tensor instead of image", "choices": None},
        {"name": "fake-vae", "flags": ["--fake-vae"], "type": "bool", "default": False, "required": False, "help": "Use deterministic fake VAE for smoke/sample tests", "choices": None},
        {"name": "use-ema", "flags": ["--use-ema", "--no-ema"], "type": "bool", "default": True, "required": False, "help": "Use EMA weights from checkpoint when available", "choices": None},
    ]


@app.get("/api/sample/args")
def get_sample_args() -> Dict[str, Any]:
    return {"items": _sample_arg_specs()}


@app.get("/api/latents/args")
def get_latent_args() -> Dict[str, Any]:
    prep_path = ROOT_DIR / "scripts" / "prepare_latents.py"
    items = parse_argparse_args(prep_path)
    defaults = _latent_prepare_arg_defaults()
    for item in items:
        name = item.get("name")
        if name in defaults:
            item["default"] = defaults[name]
    return {"items": items}


@app.get("/api/samples")
def list_samples(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=500, ge=1, le=5000),
) -> Dict[str, Any]:
    items = sorted([str(p) for p in RUNS_DIR.rglob("*.png")])
    return {"items": items[offset:offset + limit], "offset": offset, "total": len(items)}


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
    if ext not in allowed_exts and content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="unsupported image format")
    safe_kind = re.sub(r"[^A-Za-z0-9_-]+", "-", kind or "image").strip("-") or "image"
    safe_name = _safe_upload_name(file.filename)
    out_dir = UPLOADS_DIR / safe_kind
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ext or ".png"
    out_path = out_dir / f"{uuid.uuid4().hex[:12]}_{Path(safe_name).stem}{suffix}"
    try:
        with out_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()
    return {
        "ok": True,
        "path": str(out_path.resolve()),
        "url": _upload_relative_url(out_path),
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
    run = job_manager.stop_current()
    if not run:
        raise HTTPException(status_code=404, detail="No running job")
    return {"run_id": run.run_id}


@app.post("/api/sample/start")
def start_sample(req: SampleRequest, _: None = Depends(_require_token)) -> Dict[str, Any]:
    try:
        run = job_manager.start_sample(req.args.to_sample_options_args())
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
    run = job_manager.stop_current()
    if not run:
        raise HTTPException(status_code=404, detail="No running job")
    return {"run_id": run.run_id}


@app.post("/api/latents/stop")
def stop_latent_cache(_: None = Depends(_require_token)) -> Dict[str, Any]:
    run = job_manager.stop_current()
    if not run:
        raise HTTPException(status_code=404, detail="No running job")
    return {"run_id": run.run_id}


@app.websocket("/ws/logs/{run_id}")
async def ws_logs(
    websocket: WebSocket,
    run_id: str,
    backlog: int = Query(default=2000, ge=0, le=50000),
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
