from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch

from diffusion.utils import load_ckpt

from .argparse_reader import parse_argparse_args
from .job_manager import JobManager
from .services import config_service


ROOT_DIR = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT_DIR / "webui_runs"


class WSManager:
    def __init__(self) -> None:
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.connections: Dict[str, Dict[str, List[WebSocket]]] = {}

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    async def connect(self, run_id: str, channel: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections.setdefault(run_id, {}).setdefault(channel, []).append(websocket)

    def disconnect(self, run_id: str, channel: str, websocket: WebSocket) -> None:
        if run_id in self.connections and channel in self.connections[run_id]:
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
job_manager = JobManager(ROOT_DIR, ws_manager)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")


def _get_out_dir() -> Path:
    cfg_path = config_service.get_config_path(ROOT_DIR)
    cfg = config_service.parse_config_text(cfg_path.read_text(encoding="utf-8"))
    out_dir = Path(cfg.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT_DIR / out_dir
    return out_dir


class SampleRequest(BaseModel):
    args: Dict[str, Any]


class TrainRequest(BaseModel):
    resume: Optional[str] = None


class LatentCacheRequest(BaseModel):
    args: Dict[str, Any]

@app.on_event("startup")
async def _startup() -> None:
    job_manager.set_loop(asyncio.get_event_loop())


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


@app.get("/api/runs/{run_id}/logs/{stream}")
def get_run_log(run_id: str, stream: str) -> Dict[str, Any]:
    try:
        run = job_manager.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if stream == "stdout":
        path = Path(run.log_stdout)
    else:
        path = Path(run.log_stderr)
    if not path.exists():
        return {"content": ""}
    return {"content": path.read_text(encoding="utf-8")}


@app.get("/api/runs/{run_id}/metrics")
def get_run_metrics(run_id: str) -> Dict[str, Any]:
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
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return {"items": items}


@app.get("/api/config")
def get_train_config() -> Dict[str, Any]:
    cfg_path = config_service.get_config_path(ROOT_DIR)
    return {"path": str(cfg_path), "content": config_service.read_config_text(ROOT_DIR)}


@app.put("/api/config")
def update_train_config(payload: Dict[str, Any]) -> Dict[str, Any]:
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
    ckpt_path = Path(path)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT_DIR / ckpt_path
    try:
        ckpt_path = ckpt_path.resolve()
        ckpt_path.relative_to(ROOT_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid checkpoint path") from exc
    if not ckpt_path.exists():
        raise HTTPException(status_code=404, detail="checkpoint not found")
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


@app.get("/api/sample/args")
def get_sample_args() -> Dict[str, Any]:
    sample_path = ROOT_DIR / "scripts" / "sample.py"
    return {"items": parse_argparse_args(sample_path)}


@app.get("/api/latents/args")
def get_latent_args() -> Dict[str, Any]:
    prep_path = ROOT_DIR / "scripts" / "prepare_latents.py"
    return {"items": parse_argparse_args(prep_path)}


@app.get("/api/samples")
def list_samples() -> Dict[str, Any]:
    items = sorted([str(p) for p in RUNS_DIR.rglob("*.png")])
    return {"items": items}


@app.post("/api/train/start")
def start_train(req: TrainRequest | None = None) -> Dict[str, Any]:
    try:
        resume = req.resume if req else None
        run = job_manager.start_train(resume=resume)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"run_id": run.run_id, "command": run.command}


@app.post("/api/train/stop")
def stop_train() -> Dict[str, Any]:
    run = job_manager.stop_current()
    if not run:
        raise HTTPException(status_code=404, detail="No running job")
    return {"run_id": run.run_id}


@app.post("/api/sample/start")
def start_sample(req: SampleRequest) -> Dict[str, Any]:
    try:
        run = job_manager.start_sample(req.args)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"run_id": run.run_id, "command": run.command, "output": run.output_path}


@app.post("/api/latents/start")
def start_latent_cache(req: LatentCacheRequest) -> Dict[str, Any]:
    try:
        run = job_manager.start_prepare_latents(req.args)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"run_id": run.run_id, "command": run.command}


@app.post("/api/sample/stop")
def stop_sample() -> Dict[str, Any]:
    run = job_manager.stop_current()
    if not run:
        raise HTTPException(status_code=404, detail="No running job")
    return {"run_id": run.run_id}


@app.post("/api/latents/stop")
def stop_latent_cache() -> Dict[str, Any]:
    run = job_manager.stop_current()
    if not run:
        raise HTTPException(status_code=404, detail="No running job")
    return {"run_id": run.run_id}


@app.websocket("/ws/logs/{run_id}")
async def ws_logs(websocket: WebSocket, run_id: str) -> None:
    await ws_manager.connect(run_id, "logs", websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(run_id, "logs", websocket)


@app.websocket("/ws/metrics/{run_id}")
async def ws_metrics(websocket: WebSocket, run_id: str) -> None:
    await ws_manager.connect(run_id, "metrics", websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(run_id, "metrics", websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("webui.backend.app:app", host="0.0.0.0", port=8000)
