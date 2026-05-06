#! /usr/bin/env python

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import subprocess
from pathlib import Path

import uvicorn

from config.loader import load_webui_config, parse_cli_overrides


def _frontend_mode(frontend_dir: Path) -> str:
    if (frontend_dir / "node_modules").exists() and shutil.which("npm"):
        return "dev"
    if (frontend_dir / "dist" / "index.html").exists():
        return "static"
    return "missing"


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск WebUI backend.")
    parser.add_argument("--config", default="", help="Config path. Defaults to configs/webui.kdl.")
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config value, e.g. --set webui.port=7861.",
    )
    parser.add_argument(
        "--print-config", action="store_true", help="Print resolved WebUI config and exit."
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--frontend", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--frontend-host", default=None)
    parser.add_argument("--frontend-port", type=int, default=None)
    args = parser.parse_args()

    cfg = load_webui_config(args.config or None, overrides=parse_cli_overrides(args.set_values))
    host = str(args.host if args.host is not None else cfg.host)
    port = int(args.port if args.port is not None else cfg.port)
    frontend_enabled = bool(
        args.frontend
        if args.frontend is not None
        else cfg.raw.get("webui", {}).get("frontend", True)
    )
    frontend_host = str(
        args.frontend_host
        if args.frontend_host is not None
        else cfg.raw.get("webui", {}).get("frontend_host", "127.0.0.1")
    )
    frontend_port = int(
        args.frontend_port
        if args.frontend_port is not None
        else cfg.raw.get("webui", {}).get("frontend_port", 5173)
    )
    if args.print_config:
        print(
            json.dumps(
                {"host": host, "port": port, "auto_open": cfg.auto_open, "raw": cfg.raw},
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        return

    if host in {"0.0.0.0", "::"} and not os.environ.get("WEBUI_AUTH_TOKEN"):
        raise RuntimeError(
            "Refusing to bind WebUI backend to a public interface without WEBUI_AUTH_TOKEN. "
            "Set WEBUI_AUTH_TOKEN or use --host 127.0.0.1 for local-only access."
        )

    frontend_proc: subprocess.Popen[str] | None = None
    if frontend_enabled:
        frontend_dir = Path(__file__).resolve().parent / "webui" / "frontend"
        if not frontend_dir.exists():
            raise RuntimeError(f"Frontend directory not found: {frontend_dir}")
        mode = _frontend_mode(frontend_dir)
        if mode == "dev":
            npm_path = shutil.which("npm")
            assert npm_path is not None
            cmd = [
                npm_path,
                "run",
                "dev",
                "--",
                "--host",
                frontend_host,
                "--port",
                str(frontend_port),
            ]
            env = os.environ.copy()
            env.setdefault("VITE_BACKEND_TARGET", f"http://127.0.0.1:{port}")
            frontend_proc = subprocess.Popen(cmd, cwd=str(frontend_dir), env=env)
            atexit.register(frontend_proc.terminate)
        elif mode == "missing":
            raise RuntimeError(
                "Frontend assets are missing. Run `npm ci && npm run build` in webui/frontend, "
                "install the package with bundled frontend/dist, or start with --no-frontend."
            )

    uvicorn.run(
        "webui.backend.app:app", host=host, port=port, reload=bool(args.reload), log_level="info"
    )


if __name__ == "__main__":
    main()
