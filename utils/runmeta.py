from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import os

import torch


def build_run_metadata(perf: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
    meta = {
        "torch_version": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": (str(torch.version.cuda) if torch.version.cuda is not None else None),
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    }
    if perf:
        meta["perf"] = dict(perf)
    try:
        import subprocess

        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(Path.cwd()), stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        meta["git_commit"] = None
    return _normalize_metadata(meta)


def _normalize_metadata(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.hex()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _normalize_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_metadata(v) for v in value]
    return str(value)
