from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .train import TrainConfig


def load_yaml(path: str) -> Dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected mapping in YAML config: {path}")
    return data


def build_train_config(data: Dict[str, Any]) -> TrainConfig:
    return TrainConfig.from_dict(data)


def load_train_config(path: str) -> TrainConfig:
    return build_train_config(load_yaml(path))
