from __future__ import annotations

import os
from pathlib import Path


def _is_webui_mode() -> bool:
    return os.environ.get("WEBUI") == "1"


def _webui_metrics_path() -> Path | None:
    run_dir = os.environ.get("WEBUI_RUN_DIR")
    if not run_dir:
        return None
    return Path(run_dir) / "metrics" / "train_metrics.jsonl"
