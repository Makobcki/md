from __future__ import annotations

import argparse
import json

from .discovery import default_config_path
from .loader import parse_cli_overrides, resolve_target_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve md-diffusion KDL configs.")
    parser.add_argument("--target", required=True, choices=("train", "sample", "webui", "eval", "cache"))
    parser.add_argument("--config", default="", help="Override config path. Defaults to target config.")
    parser.add_argument("--set", dest="set_values", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--sources", action="store_true", help="Include config/preset source paths.")
    args = parser.parse_args()

    path = args.config or default_config_path(args.target)
    resolved = resolve_target_config(
        args.target,
        path,
        overrides=parse_cli_overrides(args.set_values),
    )
    payload: dict[str, object] = {
        "target": resolved.target,
        "version": resolved.version,
        "config": resolved.data,
    }
    if args.sources:
        payload["sources"] = [str(path) for path in resolved.sources]
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
