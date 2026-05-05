from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainRunPaths:
    base_dir: Path
    run_dir: Path
    checkpoints_dir: Path
    samples_dir: Path
    eval_dir: Path
    events_path: Path
    train_log_path: Path
    cache_manifest_path: Path


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_")


def make_train_run_dir(base_dir: str | Path, *, run_name: str = "", now: float | None = None) -> Path:
    base = Path(base_dir)
    stamp = time.strftime("%Y-%m-%d_%H%M%S", time.localtime(now or time.time()))
    suffix = _safe_name(run_name)
    prefix = f"{stamp}_{suffix}" if suffix else stamp
    idx = 1
    while True:
        candidate = base / f"{prefix}_{idx:03d}"
        if not candidate.exists():
            return candidate
        idx += 1


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def prepare_train_run_structure(
    *,
    base_out_dir: str | Path,
    cfg_dict: dict[str, Any],
    run_name: str = "",
    cache_manifest_source: str | Path | None = None,
    create_unique_subdir: bool = True,
) -> TrainRunPaths:
    base_dir = Path(base_out_dir)
    run_dir = make_train_run_dir(base_dir, run_name=run_name) if create_unique_subdir else base_dir
    checkpoints_dir = run_dir / "checkpoints"
    samples_dir = run_dir / "samples"
    eval_dir = run_dir / "eval"
    for path in (checkpoints_dir, samples_dir, eval_dir):
        path.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    train_log_path = run_dir / "train.log"
    train_log_path.touch(exist_ok=True)
    # config.yaml is the canonical resolved training config for this run.
    _write_yaml(run_dir / "config.yaml", cfg_dict)
    # Compatibility aliases for older UI/readers; kept explicit in config_manifest.yaml.
    _write_yaml(run_dir / "config_resolved.yaml", cfg_dict)
    _write_yaml(run_dir / "config_snapshot.yaml", cfg_dict)
    _write_yaml(run_dir / "config_manifest.yaml", {
        "canonical": "config.yaml",
        "aliases": ["config_resolved.yaml", "config_snapshot.yaml"],
    })
    cache_manifest_path = run_dir / "cache_manifest.json"
    if cache_manifest_source is not None and Path(cache_manifest_source).exists():
        cache_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache_manifest_source, cache_manifest_path)
    else:
        cache_manifest_path.write_text("{}\n", encoding="utf-8")
    return TrainRunPaths(
        base_dir=base_dir,
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        samples_dir=samples_dir,
        eval_dir=eval_dir,
        events_path=run_dir / "events.jsonl",
        train_log_path=train_log_path,
        cache_manifest_path=cache_manifest_path,
    )
