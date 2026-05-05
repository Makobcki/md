from __future__ import annotations

import asyncio
import json
import os
import signal
import contextlib
import subprocess
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.train import TrainConfig
from diffusion.events import format_event_line
from sample.api import SampleOptions, run_sample
from .services import config_service
from .services.atomic import atomic_write_json, atomic_write_text


METRIC_EVENT_TYPES = {"metric", "progress", "train", "eval", "sample"}


def is_metric_event(event: dict | None) -> bool:
    return isinstance(event, dict) and event.get("type") in METRIC_EVENT_TYPES


@dataclass
class RunRecord:
    run_id: str
    run_type: str
    status: str
    command: List[str]
    created_at: str
    pid: Optional[int]
    started_at: str
    ended_at: Optional[str]
    exit_code: Optional[int]
    run_dir: str
    config_snapshot: Optional[str]
    log_stdout: str
    log_stderr: str
    metrics_path: Optional[str]
    output_path: Optional[str]
    notes: Dict[str, Any]


class JobManager:
    def __init__(self, repo_root: Path, ws_manager: Any, runs_dir: Path | None = None) -> None:
        self.repo_root = repo_root
        self.runs_dir = runs_dir or (repo_root / "webui_runs")
        self.state_path = self.runs_dir / "state.json"
        self.ws_manager = ws_manager
        self.lock = threading.Lock()
        self.process: Optional[subprocess.Popen] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.metric_tailers: Dict[str, tuple[threading.Event, threading.Thread]] = {}
        self.current_run_id: Optional[str] = None
        self.runs: Dict[str, RunRecord] = {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._load_state()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.ws_manager.set_loop(loop)

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        for item in data.get("runs", []):
            if "pid" not in item:
                item["pid"] = None
            self.runs[item["run_id"]] = RunRecord(**item)
        changed = False
        recovered_active = False
        for run in self.runs.values():
            if run.status not in {"queued", "running", "stopping"}:
                continue
            if run.pid and self._pid_alive(run.pid):
                run.notes = dict(run.notes or {})
                run.notes["recovered_after_backend_restart"] = True
                run.notes["recovery_mode"] = "degraded_no_log_or_metric_tailers"
                if not recovered_active:
                    run.status = "running"
                    self.current_run_id = run.run_id
                    recovered_active = True
                else:
                    run.status = "recovered"
                changed = True
            else:
                run.status = "interrupted"
                run.exit_code = -1
                run.ended_at = time.strftime("%Y-%m-%dT%H:%M:%S")
                run.pid = None
                changed = True
        if changed:
            self._save_state()

    def _save_state(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        data = {"runs": [asdict(r) for r in self.runs.values()]}
        atomic_write_json(self.state_path, data)

    def _new_run_id(self) -> str:
        return f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def _pid_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _write_notes(self, run_dir: Path, notes: Dict[str, Any]) -> None:
        notes_path = run_dir / "notes.json"
        atomic_write_json(notes_path, notes)

    @staticmethod
    def _parse_event_line(line: str) -> dict | None:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None

    @classmethod
    def format_log_line(cls, line: str) -> str:
        event = cls._parse_event_line(line)
        if event is None or not event.get("type"):
            return line
        return format_event_line(event)

    def _read_train_config(self) -> Dict[str, Any]:
        cfg_path = config_service.get_config_path(self.repo_root)
        return TrainConfig.from_yaml(str(cfg_path)).to_dict()

    def _allowed_output_roots(self) -> list[Path]:
        roots = [self.repo_root, self.runs_dir]
        try:
            out_dir = Path(self._read_train_config().get("out_dir", "./runs"))
            if not out_dir.is_absolute():
                out_dir = self.repo_root / out_dir
            roots.append(out_dir)
        except Exception:
            pass
        for item in os.environ.get("WEBUI_ALLOWED_PATHS", "").split(os.pathsep):
            item = item.strip()
            if item:
                roots.append(Path(item))
        return [root.resolve() for root in roots]

    def _is_allowed_output_path(self, path: Path) -> bool:
        resolved = path.resolve()
        return any(resolved == root or resolved.is_relative_to(root) for root in self._allowed_output_roots())

    def list_runs(self) -> List[RunRecord]:
        return sorted(self.runs.values(), key=lambda r: r.created_at, reverse=True)

    def get_run(self, run_id: str) -> RunRecord:
        if run_id not in self.runs:
            raise KeyError(run_id)
        return self.runs[run_id]

    def _has_active_job_locked(self) -> bool:
        if not self.current_run_id:
            return False
        run = self.runs.get(self.current_run_id)
        if run is None:
            self.current_run_id = None
            return False
        if self.process is not None:
            return True
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return True
        if run.pid and self._pid_alive(run.pid):
            return True
        return run.status in {"queued", "running", "stopping"}

    def _ensure_no_active_job_locked(self) -> None:
        if self._has_active_job_locked():
            raise RuntimeError("Another job is running.")

    def get_status(self) -> Dict[str, Any]:
        if not self.current_run_id:
            return {"active": False}
        run = self.runs[self.current_run_id]
        if self.process is None and (self.worker_thread is None or not self.worker_thread.is_alive()):
            if run.pid and not self._pid_alive(run.pid):
                run.status = "failed"
                run.exit_code = -1
                run.ended_at = time.strftime("%Y-%m-%dT%H:%M:%S")
                run.pid = None
                self.current_run_id = None
                self._save_state()
                return {"active": False}
            if run.status in {"queued", "running", "stopping"}:
                run.status = "failed"
                run.exit_code = -1
                run.ended_at = time.strftime("%Y-%m-%dT%H:%M:%S")
                self.current_run_id = None
                self._save_state()
                return {"active": False}
        return {"active": True, "run": asdict(run)}

    def list_checkpoints(self) -> List[str]:
        cfg = self._read_train_config()
        out_dir = Path(cfg.get("out_dir", "./runs"))
        if not out_dir.is_absolute():
            out_dir = self.repo_root / out_dir
        if not out_dir.exists():
            return []
        candidates: list[Path] = []
        checkpoint_dir = out_dir / "checkpoints"
        if checkpoint_dir.exists():
            candidates.extend(checkpoint_dir.glob("*.pt"))
        candidates.extend(out_dir.glob("ckpt_*.pt"))
        candidates.extend(out_dir.glob("step_*.pt"))
        return sorted({str(p) for p in candidates if p.is_file() or p.is_symlink()})

    def _emit_metric_line_from_file(self, run: RunRecord, line: str) -> None:
        event = self._parse_event_line(line)
        if event is not None and is_metric_event(event):
            self.ws_manager.send_from_thread(run.run_id, "metrics", json.dumps(event, ensure_ascii=False))

    def _start_metrics_tailer(self, run: RunRecord) -> None:
        if not run.metrics_path:
            return
        existing = self.metric_tailers.get(run.run_id)
        if existing is not None and existing[1].is_alive():
            return
        stop_event = threading.Event()

        def _tail() -> None:
            path = Path(run.metrics_path or "")
            position = 0

            def _drain_available() -> None:
                nonlocal position
                if not path.exists():
                    return
                try:
                    with path.open("r", encoding="utf-8") as f:
                        f.seek(position)
                        while True:
                            line = f.readline()
                            if not line:
                                position = f.tell()
                                return
                            position = f.tell()
                            self._emit_metric_line_from_file(run, line.rstrip("\n"))
                except OSError:
                    return

            while not stop_event.is_set():
                _drain_available()
                stop_event.wait(0.25)
            # One final drain after the process exits so events flushed by the
            # training-side async event writer are not missed.
            deadline = time.time() + 2.0
            while time.time() < deadline:
                before = position
                _drain_available()
                if position == before:
                    break
                time.sleep(0.05)

        thread = threading.Thread(target=_tail, name=f"metrics-tail-{run.run_id}", daemon=True)
        self.metric_tailers[run.run_id] = (stop_event, thread)
        thread.start()

    def _stop_metrics_tailer(self, run_id: str) -> None:
        item = self.metric_tailers.pop(run_id, None)
        if item is None:
            return
        stop_event, thread = item
        stop_event.set()
        thread.join(timeout=3.0)

    def _start_process(self, run: RunRecord, env: Dict[str, str]) -> None:
        run_dir = Path(run.run_dir)
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        stdout_path = Path(run.log_stdout)
        stderr_path = Path(run.log_stderr)
        stdout_f = stdout_path.open("a", encoding="utf-8")
        stderr_f = stderr_path.open("a", encoding="utf-8")

        proc = subprocess.Popen(
            run.command,
            cwd=str(self.repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
            start_new_session=True,
        )
        self.process = proc
        self.current_run_id = run.run_id
        run.status = "running"
        run.pid = proc.pid
        run.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._save_state()
        self._start_metrics_tailer(run)

        def _reader(stream, stream_name: str) -> None:
            for line in iter(stream.readline, ""):
                line = line.rstrip("\n")
                event = self._parse_event_line(line)
                log_line = format_event_line(event) if event is not None and event.get("type") else line
                if stream_name == "stdout":
                    stdout_f.write(log_line + "\n")
                    stdout_f.flush()
                else:
                    stderr_f.write(log_line + "\n")
                    stderr_f.flush()

                self.ws_manager.send_from_thread(run.run_id, "logs", json.dumps({
                    "type": "log",
                    "stream": stream_name,
                    "line": log_line,
                }, ensure_ascii=False))
                if stream_name == "stdout" and event is not None and is_metric_event(event):
                    if run.run_type == "train" and isinstance(event.get("path"), str):
                        with self.lock:
                            run.notes = dict(run.notes or {})
                            event_path = Path(str(event.get("path")))
                            if event_path.is_absolute():
                                run.notes.setdefault("latest_artifact_dir", str(event_path.parent))
                            elif run.notes.get("train_run_dir"):
                                run.notes.setdefault("latest_artifact_dir", str((Path(str(run.notes["train_run_dir"])) / event_path).parent))
                            self._save_state()
                    self.ws_manager.send_from_thread(run.run_id, "metrics", json.dumps(event, ensure_ascii=False))

        stdout_thread = threading.Thread(target=_reader, args=(proc.stdout, "stdout"), daemon=True)
        stderr_thread = threading.Thread(target=_reader, args=(proc.stderr, "stderr"), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        def _waiter() -> None:
            proc.wait()
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            self._stop_metrics_tailer(run.run_id)
            stdout_f.close()
            stderr_f.close()
            with self.lock:
                run.exit_code = proc.returncode
                run.ended_at = time.strftime("%Y-%m-%dT%H:%M:%S")
                run.pid = None
                if run.status == "stopping":
                    run.status = "stopped"
                elif proc.returncode == 0:
                    run.status = "done"
                else:
                    run.status = "failed"
                if self.process is proc:
                    self.process = None
                if self.current_run_id == run.run_id:
                    self.current_run_id = None
                status = run.status
                self._save_state()
            self.ws_manager.send_from_thread(run.run_id, "logs", json.dumps({
                "type": "status",
                "status": status,
                "exit_code": proc.returncode,
            }, ensure_ascii=False))

        threading.Thread(target=_waiter, daemon=True).start()

    def start_train(self, resume: Optional[str] = None) -> RunRecord:
        with self.lock:
            self._ensure_no_active_job_locked()
            run_id = self._new_run_id()
            run_dir = self.runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_dir = run_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)

            config_snapshot = run_dir / "config_snapshot.yaml"
            cfg_path = config_service.get_config_path(self.repo_root)
            atomic_write_text(config_snapshot, cfg_path.read_text(encoding="utf-8"))

            cmd = [
                os.environ.get("PYTHON", sys.executable),
                "-u",
                "-m",
                "train.cli",
                "--config",
                str(cfg_path),
            ]
            if resume:
                cmd.extend(["--resume", str(resume)])

            run = RunRecord(
                run_id=run_id,
                run_type="train",
                status="queued",
                command=cmd,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                pid=None,
                started_at="",
                ended_at=None,
                exit_code=None,
                run_dir=str(run_dir),
                config_snapshot=str(config_snapshot),
                log_stdout=str(run_dir / "logs" / "train.log"),
                log_stderr=str(run_dir / "logs" / "train.err.log"),
                metrics_path=str(metrics_dir / "train_metrics.jsonl"),
                output_path=None,
                notes={},
            )
            self.runs[run_id] = run
            self._save_state()
            notes: Dict[str, Any] = {"type": "train"}
            try:
                cfg_dict = self._read_train_config()
                out_dir = Path(str(cfg_dict.get("out_dir", "./runs")))
                if not out_dir.is_absolute():
                    out_dir = self.repo_root / out_dir
                notes["out_dir"] = str(out_dir.resolve())
                run.notes = dict(run.notes or {}) | notes
            except Exception:
                pass
            if resume:
                notes["resume"] = str(resume)
            self._write_notes(run_dir, notes)

            env = os.environ.copy()
            env["WEBUI"] = "1"
            env["WEBUI_RUN_DIR"] = str(run_dir)
            env["PYTHONUNBUFFERED"] = "1"

            repo_root_str = str(self.repo_root)
            env["PYTHONPATH"] = repo_root_str + os.pathsep + env.get("PYTHONPATH", "")

            self._start_process(run, env)
            return run

    class _ThreadLogWriter:
        def __init__(self, manager: "JobManager", run: RunRecord, stream_name: str, file_obj: Any) -> None:
            self.manager = manager
            self.run = run
            self.stream_name = stream_name
            self.file_obj = file_obj
            self.buffer = ""

        def write(self, data: str) -> int:
            if not data:
                return 0
            self.buffer += data
            while "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                self._emit(line.rstrip("\r"))
            return len(data)

        def flush(self) -> None:
            if self.buffer:
                self._emit(self.buffer.rstrip("\r"))
                self.buffer = ""
            self.file_obj.flush()

        def _emit(self, line: str) -> None:
            event = self.manager._parse_event_line(line)
            log_line = format_event_line(event) if event is not None and event.get("type") else line
            self.file_obj.write(log_line + "\n")
            self.file_obj.flush()
            self.manager.ws_manager.send_from_thread(
                self.run.run_id,
                "logs",
                json.dumps({"type": "log", "stream": self.stream_name, "line": log_line}, ensure_ascii=False),
            )
            if self.stream_name == "stdout" and event is not None and is_metric_event(event):
                self.manager.ws_manager.send_from_thread(self.run.run_id, "metrics", line)

    @staticmethod
    def _normalize_sample_args(args: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        aliases = {
            "negative-prompt": "neg_prompt",
            "neg-prompt": "neg_prompt",
            "init-image": "init_image",
            "control-image": "control_image",
            "control-strength": "control_strength",
            "control-type": "control_type",
            "latent-only": "latent_only",
            "fake-vae": "fake_vae",
            "use-ema": "use_ema",
            "no-ema": "use_ema",
        }
        for key, value in args.items():
            name = aliases.get(str(key), str(key).replace("-", "_"))
            if str(key) == "no-ema":
                value = not bool(value)
            if value is None:
                continue
            if name == "device" and value == "auto":
                continue
            normalized[name] = value
        return normalized

    def _start_sample_thread(self, run: RunRecord, options: SampleOptions) -> None:
        run_dir = Path(run.run_dir)
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = Path(run.log_stdout)
        stderr_path = Path(run.log_stderr)
        stdout_f = stdout_path.open("a", encoding="utf-8")
        stderr_f = stderr_path.open("a", encoding="utf-8")

        def _write_event(event: dict[str, object]) -> None:
            line = json.dumps(event, ensure_ascii=False)
            log_line = format_event_line(event) if event.get("type") else line
            stdout_f.write(log_line + "\n")
            stdout_f.flush()
            self.ws_manager.send_from_thread(
                run.run_id,
                "logs",
                json.dumps({"type": "log", "stream": "stdout", "line": log_line}, ensure_ascii=False),
            )
            if is_metric_event(event):
                self.ws_manager.send_from_thread(run.run_id, "metrics", line)

        def _worker() -> None:
            exit_code = 0
            try:
                run_sample(options, event_callback=_write_event, quiet=True)
            except Exception:
                exit_code = 1
                stderr_f.write(traceback.format_exc() + "\n")
                stderr_f.flush()
            finally:
                stdout_f.close()
                stderr_f.close()
                with self.lock:
                    run.exit_code = exit_code
                    run.ended_at = time.strftime("%Y-%m-%dT%H:%M:%S")
                    run.pid = None
                    if run.status == "stopping":
                        run.status = "stopped"
                    elif exit_code == 0:
                        run.status = "done"
                    else:
                        run.status = "failed"
                    if self.current_run_id == run.run_id:
                        self.current_run_id = None
                    self.worker_thread = None
                    self._save_state()
                self.ws_manager.send_from_thread(
                    run.run_id,
                    "logs",
                    json.dumps({"type": "status", "status": run.status, "exit_code": exit_code}, ensure_ascii=False),
                )

        run.status = "running"
        run.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.current_run_id = run.run_id
        thread = threading.Thread(target=_worker, name=f"sample-api-{run.run_id}", daemon=True)
        self.worker_thread = thread
        self._save_state()
        thread.start()

    def start_sample(self, args: Dict[str, Any]) -> RunRecord:
        with self.lock:
            self._ensure_no_active_job_locked()
            run_id = self._new_run_id()
            run_dir = self.runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            samples_dir = run_dir / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)

            normalized_args = self._normalize_sample_args(dict(args))
            output_path = normalized_args.get("out")
            notes: Dict[str, Any] = {"args": normalized_args, "backend": "sample.api"}
            if not output_path:
                suffix = ".pt" if bool(normalized_args.get("latent_only")) else ".png"
                output_path = samples_dir / f"sample{suffix}"
            else:
                output_path = Path(str(output_path))
                if output_path.is_absolute():
                    resolved = output_path.resolve()
                    if self._is_allowed_output_path(resolved):
                        output_path = resolved
                    else:
                        output_path = samples_dir / output_path.name
                        notes["warning"] = "output path outside repo; rewritten into run samples"
                else:
                    output_path = samples_dir / output_path.name
            normalized_args["out"] = str(output_path)

            options = SampleOptions(**normalized_args)
            options.validate()
            use_legacy_thread = not self.repo_root.exists()
            command = ["sample.api.run_sample"] if use_legacy_thread else [os.environ.get("PYTHON", sys.executable), "-u", "-m", "sample.cli"]
            cli_flags = {
                "neg_prompt": "--neg_prompt",
                "init_image": "--init-image",
                "control_image": "--control-image",
                "control_strength": "--control-strength",
                "control_type": "--control-type",
                "latent_only": "--latent-only",
                "fake_vae": "--fake-vae",
                "use_ema": "--use-ema",
            }
            if not use_legacy_thread:
                for key, value in normalized_args.items():
                    flag = cli_flags.get(key, f"--{key.replace('_', '-')}")
                    if isinstance(value, bool):
                        if key == "use_ema":
                            command.append("--use-ema" if value else "--no-ema")
                        elif value:
                            command.append(flag)
                    elif value is not None and value != "":
                        command.extend([flag, str(value)])

            run = RunRecord(
                run_id=run_id,
                run_type="sample",
                status="queued",
                command=command,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                pid=None,
                started_at="",
                ended_at=None,
                exit_code=None,
                run_dir=str(run_dir),
                config_snapshot=None,
                log_stdout=str(run_dir / "logs" / "sample.log"),
                log_stderr=str(run_dir / "logs" / "sample.err.log"),
                metrics_path=None,
                output_path=str(output_path),
                notes=notes,
            )
            self.runs[run_id] = run
            self._save_state()
            self._write_notes(run_dir, {"type": "sample", "args": normalized_args, "output": str(output_path), **notes})

            if use_legacy_thread:
                notes["warning"] = "sample subprocess disabled because repo_root does not exist"
                self._start_sample_thread(run, options)
                return run

            env = os.environ.copy()
            env["WEBUI"] = "1"
            env["WEBUI_RUN_DIR"] = str(run_dir)
            env["PYTHONUNBUFFERED"] = "1"
            repo_root_str = str(self.repo_root)
            env["PYTHONPATH"] = repo_root_str + os.pathsep + env.get("PYTHONPATH", "")
            self._start_process(run, env)
            return run
    def start_prepare_latents(self, args: Dict[str, Any]) -> RunRecord:
        with self.lock:
            self._ensure_no_active_job_locked()
            run_id = self._new_run_id()
            run_dir = self.runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            logs_dir = run_dir / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                os.environ.get("PYTHON", sys.executable),
                "-u",
                "scripts/prepare_latents.py",
            ]
            for key, value in args.items():
                flag = f"--{key}"
                if isinstance(value, bool):
                    if value:
                        cmd.append(flag)
                    elif key == "pin-memory":
                        cmd.append("--no-pin-memory")
                    elif key == "overwrite":
                        cmd.append("--no-overwrite")
                else:
                    cmd.extend([flag, str(value)])

            run = RunRecord(
                run_id=run_id,
                run_type="latent_cache",
                status="queued",
                command=cmd,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                pid=None,
                started_at="",
                ended_at=None,
                exit_code=None,
                run_dir=str(run_dir),
                config_snapshot=None,
                log_stdout=str(logs_dir / "latents.log"),
                log_stderr=str(logs_dir / "latents.err.log"),
                metrics_path=str(run_dir / "metrics" / "latent_prepare.jsonl"),
                output_path=None,
                notes={"type": "latent_cache", "args": args},
            )
            self.runs[run_id] = run
            self._save_state()
            self._write_notes(run_dir, {"type": "latent_cache", "args": args})

            env = os.environ.copy()
            env["WEBUI"] = "1"
            env["WEBUI_RUN_DIR"] = str(run_dir)
            env["PYTHONUNBUFFERED"] = "1"

            # чтобы импортировался локальный пакет repo_root/diffusion
            repo_root_str = str(self.repo_root)
            env["PYTHONPATH"] = repo_root_str + os.pathsep + env.get("PYTHONPATH", "")

            self._start_process(run, env)
            return run

    def stop_current(
        self,
        expected_type: str | None = None,
        timeout_int: int = 8,
        timeout_term: int = 5,
        timeout_kill: int = 2,
    ) -> Optional[RunRecord]:
        def _send(pid: int, sig: signal.Signals) -> None:
            try:
                os.killpg(pid, sig)
            except ProcessLookupError:
                return
            except OSError:
                os.kill(pid, sig)

        proc: subprocess.Popen | None = None
        pid: int | None = None
        run: RunRecord | None = None
        with self.lock:
            if self.current_run_id is None:
                return None
            run = self.runs.get(self.current_run_id)
            if run is None:
                self.current_run_id = None
                self._save_state()
                return None
            if expected_type is not None and run.run_type != expected_type:
                raise RuntimeError(f"Active job is {run.run_type}, not {expected_type}.")
            run.status = "stopping"
            proc = self.process
            pid = proc.pid if proc is not None else run.pid
            self._save_state()
            if proc is None and self.worker_thread is not None and self.worker_thread.is_alive():
                # Thread-only legacy jobs cannot be force-stopped. New sample jobs
                # run as subprocesses; this branch only marks old recovered jobs.
                return run

        if pid is None:
            return run

        for sig, timeout in (
            (signal.SIGINT, timeout_int),
            (signal.SIGTERM, timeout_term),
            (signal.SIGKILL, timeout_kill),
        ):
            try:
                _send(pid, sig)
            except OSError:
                break
            deadline = time.time() + timeout
            while time.time() < deadline:
                if proc is not None:
                    if proc.poll() is not None:
                        return run
                elif not self._pid_alive(pid):
                    with self.lock:
                        if run is not None:
                            run.status = "stopped"
                            run.exit_code = 0 if sig is not signal.SIGKILL else -signal.SIGKILL
                            run.ended_at = time.strftime("%Y-%m-%dT%H:%M:%S")
                            run.pid = None
                        if self.current_run_id == (run.run_id if run else None):
                            self.current_run_id = None
                        self._save_state()
                    return run
                time.sleep(0.2)
        return run
