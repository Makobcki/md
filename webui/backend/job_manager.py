from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.train import TrainConfig
from .services import config_service


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
    def __init__(self, repo_root: Path, ws_manager: Any) -> None:
        self.repo_root = repo_root
        self.runs_dir = repo_root / "webui_runs"
        self.state_path = self.runs_dir / "state.json"
        self.ws_manager = ws_manager
        self.lock = threading.Lock()
        self.process: Optional[subprocess.Popen] = None
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
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        for item in data.get("runs", []):
            if "pid" not in item:
                item["pid"] = None
            self.runs[item["run_id"]] = RunRecord(**item)
        for run in self.runs.values():
            if run.status == "running" and run.pid and self._pid_alive(run.pid):
                self.current_run_id = run.run_id
                break

    def _save_state(self) -> None:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        data = {"runs": [asdict(r) for r in self.runs.values()]}
        self.state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

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
        notes_path.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_train_config(self) -> Dict[str, Any]:
        cfg_path = config_service.get_config_path(self.repo_root)
        return TrainConfig.from_yaml(str(cfg_path)).to_dict()

    def list_runs(self) -> List[RunRecord]:
        return sorted(self.runs.values(), key=lambda r: r.created_at, reverse=True)

    def get_run(self, run_id: str) -> RunRecord:
        if run_id not in self.runs:
            raise KeyError(run_id)
        return self.runs[run_id]

    def get_status(self) -> Dict[str, Any]:
        if not self.current_run_id:
            return {"active": False}
        run = self.runs[self.current_run_id]
        if run.pid and not self._pid_alive(run.pid) and self.process is None:
            run.status = "failed"
            run.exit_code = -1
            run.ended_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            run.pid = None
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
        return sorted([str(p) for p in out_dir.rglob("*.pt")])

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
        )
        self.process = proc
        self.current_run_id = run.run_id
        run.status = "running"
        run.pid = proc.pid
        run.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._save_state()

        def _reader(stream, stream_name: str) -> None:
            for line in iter(stream.readline, ""):
                line = line.rstrip("\n")
                if stream_name == "stdout":
                    stdout_f.write(line + "\n")
                    stdout_f.flush()
                else:
                    stderr_f.write(line + "\n")
                    stderr_f.flush()

                self.ws_manager.send_from_thread(run.run_id, "logs", json.dumps({
                    "type": "log",
                    "stream": stream_name,
                    "line": line,
                }, ensure_ascii=False))

                if stream_name == "stdout":
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        obj = None
                    if isinstance(obj, dict) and obj.get("type") == "metric":
                        self.ws_manager.send_from_thread(run.run_id, "metrics", line)

        stdout_thread = threading.Thread(target=_reader, args=(proc.stdout, "stdout"), daemon=True)
        stderr_thread = threading.Thread(target=_reader, args=(proc.stderr, "stderr"), daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        def _waiter() -> None:
            proc.wait()
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            stdout_f.close()
            stderr_f.close()
            run.exit_code = proc.returncode
            run.ended_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            run.pid = None
            if run.status == "stopping" and proc.returncode == 0:
                run.status = "stopped"
            elif proc.returncode == 0:
                run.status = "done"
            else:
                run.status = "failed"
            self.process = None
            self.current_run_id = None
            self._save_state()
            self.ws_manager.send_from_thread(run.run_id, "logs", json.dumps({
                "type": "status",
                "status": run.status,
                "exit_code": proc.returncode,
            }, ensure_ascii=False))

        threading.Thread(target=_waiter, daemon=True).start()

    def start_train(self, resume: Optional[str] = None) -> RunRecord:
        with self.lock:
            if self.process is not None:
                raise RuntimeError("Another job is running.")
            if self.current_run_id:
                run = self.runs[self.current_run_id]
                if run.pid and self._pid_alive(run.pid):
                    raise RuntimeError("Another job is running.")
            run_id = self._new_run_id()
            run_dir = self.runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            metrics_dir = run_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)

            config_snapshot = run_dir / "config_snapshot.yaml"
            cfg_path = self.repo_root / "config" / "train.yaml"
            config_snapshot.write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")

            cmd = [
                os.environ.get("PYTHON", sys.executable),
                "-u",
                "-m",
                "train.cli",
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

    def start_sample(self, args: Dict[str, Any]) -> RunRecord:
        with self.lock:
            if self.process is not None:
                raise RuntimeError("Another job is running.")
            if self.current_run_id:
                run = self.runs[self.current_run_id]
                if run.pid and self._pid_alive(run.pid):
                    raise RuntimeError("Another job is running.")
            run_id = self._new_run_id()
            run_dir = self.runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            samples_dir = run_dir / "samples"
            samples_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                os.environ.get("PYTHON", sys.executable),
                "-u",
                "-m",
                "sample",
            ]
            output_path = args.get("out")
            notes: Dict[str, Any] = {"args": args}
            if not output_path:
                output_path = samples_dir / "sample.png"
            else:
                output_path = Path(output_path)
                if output_path.is_absolute():
                    resolved = output_path.resolve()
                    try:
                        resolved.relative_to(self.repo_root.resolve())
                    except ValueError:
                        output_path = samples_dir / output_path.name
                        notes["warning"] = "output path outside repo; rewritten into run samples"
                    else:
                        output_path = resolved
                else:
                    output_path = samples_dir / output_path.name
            args["out"] = str(output_path)

            for key, value in args.items():
                flag = f"--{key}"
                if isinstance(value, bool):
                    if value:
                        cmd.append(flag)
                else:
                    cmd.extend([flag, str(value)])

            run = RunRecord(
                run_id=run_id,
                run_type="sample",
                status="queued",
                command=cmd,
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
            self._write_notes(run_dir, {"type": "sample", "args": args, "output": str(output_path), **notes})

            env = os.environ.copy()
            env["WEBUI"] = "1"
            env["WEBUI_RUN_DIR"] = str(run_dir)
            env["PYTHONUNBUFFERED"] = "1"

            # чтобы импортировался локальный пакет repo_root/diffusion
            repo_root_str = str(self.repo_root)
            env["PYTHONPATH"] = repo_root_str + os.pathsep + env.get("PYTHONPATH", "")

            self._start_process(run, env)
            return run
    def start_prepare_latents(self, args: Dict[str, Any]) -> RunRecord:
        with self.lock:
            if self.process is not None:
                raise RuntimeError("Another job is running.")
            if self.current_run_id:
                run = self.runs[self.current_run_id]
                if run.pid and self._pid_alive(run.pid):
                    raise RuntimeError("Another job is running.")
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
                metrics_path=None,
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

    def stop_current(self, timeout_int: int = 8, timeout_term: int = 5) -> Optional[RunRecord]:
        with self.lock:
            if self.process is None or self.current_run_id is None:
                if self.current_run_id:
                    run = self.runs[self.current_run_id]
                    if run.pid and self._pid_alive(run.pid):
                        run.status = "stopping"
                        self._save_state()
                        try:
                            os.kill(run.pid, signal.SIGINT)
                        except OSError:
                            return run
                        deadline = time.time() + timeout_int
                        while time.time() < deadline:
                            if not self._pid_alive(run.pid):
                                run.status = "stopped"
                                run.exit_code = 0
                                run.ended_at = time.strftime("%Y-%m-%dT%H:%M:%S")
                                run.pid = None
                                self.current_run_id = None
                                self._save_state()
                                return run
                            time.sleep(0.2)
                        try:
                            os.kill(run.pid, signal.SIGTERM)
                        except OSError:
                            return run
                        deadline = time.time() + timeout_term
                        while time.time() < deadline:
                            if not self._pid_alive(run.pid):
                                run.status = "stopped"
                                run.exit_code = 0
                                run.ended_at = time.strftime("%Y-%m-%dT%H:%M:%S")
                                run.pid = None
                                self.current_run_id = None
                                self._save_state()
                                return run
                            time.sleep(0.2)
                        try:
                            os.kill(run.pid, signal.SIGKILL)
                        except OSError:
                            return run
                        return run
                return None
            run = self.runs[self.current_run_id]
            run.status = "stopping"
            self._save_state()


            proc = self.process
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=timeout_int)
                return run
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=timeout_term)
                    return run
                except subprocess.TimeoutExpired:
                    proc.kill()
                    return run
