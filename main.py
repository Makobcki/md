from __future__ import annotations

import argparse
import atexit
import shutil
import subprocess
from pathlib import Path

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск WebUI backend.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--frontend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frontend-host", default="0.0.0.0")
    parser.add_argument("--frontend-port", type=int, default=5173)
    args = parser.parse_args()

    frontend_proc: subprocess.Popen[str] | None = None
    if bool(args.frontend):
        npm_path = shutil.which("npm")
        if npm_path is None:
            raise RuntimeError("npm is required to run the frontend (npm not found).")
        frontend_dir = Path(__file__).resolve().parent / "webui" / "frontend"
        if not frontend_dir.exists():
            raise RuntimeError(f"Frontend directory not found: {frontend_dir}")
        cmd = [
            npm_path,
            "run",
            "dev",
            "--",
            "--host",
            str(args.frontend_host),
            "--port",
            str(args.frontend_port),
        ]
        frontend_proc = subprocess.Popen(cmd, cwd=str(frontend_dir))
        atexit.register(frontend_proc.terminate)

    uvicorn.run(
        "webui.backend.app:app",
        host=str(args.host),
        port=int(args.port),
        reload=bool(args.reload),
        log_level="info",
    )


if __name__ == "__main__":
    main()
