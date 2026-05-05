#! /usr/bin/env python

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import subprocess
from pathlib import Path

import uvicorn


def _frontend_mode(frontend_dir: Path) -> str:
    if (frontend_dir / "node_modules").exists() and shutil.which("npm"):
        return "dev"
    if (frontend_dir / "dist" / "index.html").exists():
        return "static"
    return "missing"


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск WebUI backend.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--frontend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frontend-host", default="0.0.0.0")
    parser.add_argument("--frontend-port", type=int, default=5173)
    args = parser.parse_args()

    if str(args.host) in {"0.0.0.0", "::"} and not os.environ.get("WEBUI_AUTH_TOKEN"):
        raise RuntimeError(
            "Refusing to bind WebUI backend to a public interface without WEBUI_AUTH_TOKEN. "
            "Set WEBUI_AUTH_TOKEN or use --host 127.0.0.1 for local-only access."
        )

    frontend_proc: subprocess.Popen[str] | None = None
    if bool(args.frontend):
        frontend_dir = Path(__file__).resolve().parent / "webui" / "frontend"
        if not frontend_dir.exists():
            raise RuntimeError(f"Frontend directory not found: {frontend_dir}")
        mode = _frontend_mode(frontend_dir)
        if mode == "dev":
            npm_path = shutil.which("npm")
            assert npm_path is not None
            cmd = [npm_path, "run", "dev", "--", "--host", str(args.frontend_host), "--port", str(args.frontend_port)]
            env = os.environ.copy()
            env.setdefault("VITE_API_BASE", f"http://127.0.0.1:{int(args.port)}")
            frontend_proc = subprocess.Popen(cmd, cwd=str(frontend_dir), env=env)
            atexit.register(frontend_proc.terminate)
        elif mode == "missing":
            raise RuntimeError(
                "Frontend assets are missing. Run `npm ci && npm run build` in webui/frontend, "
                "install the package with bundled frontend/dist, or start with --no-frontend."
            )

    uvicorn.run("webui.backend.app:app", host=str(args.host), port=int(args.port), reload=bool(args.reload), log_level="info")


if __name__ == "__main__":
    main()
