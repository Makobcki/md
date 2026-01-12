from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск WebUI backend.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "webui.backend.app:app",
        host=str(args.host),
        port=int(args.port),
        reload=bool(args.reload),
        log_level="info",
    )


if __name__ == "__main__":
    main()
