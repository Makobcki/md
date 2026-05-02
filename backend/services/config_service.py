from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict
import os

import yaml

from config.train import TrainConfig
from .atomic import atomic_write_text


def get_config_path(repo_root: Path) -> Path:
    override = os.environ.get("WEBUI_CONFIG_PATH")
    if override:
        return Path(override)
    return repo_root / "config" / "train.yaml"


def read_config_text(repo_root: Path) -> str:
    path = get_config_path(repo_root)
    return path.read_text(encoding="utf-8")


def parse_config_text(text: str) -> TrainConfig:
    data = yaml.safe_load(text)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("config must be a YAML mapping")
    return TrainConfig.from_dict(data)


def validate_config_text(text: str) -> Dict[str, Any]:
    cfg = parse_config_text(text)
    return asdict(cfg)


def write_config_text(repo_root: Path, text: str) -> Dict[str, Any]:
    cfg_dict = validate_config_text(text)
    path = get_config_path(repo_root)
    atomic_write_text(path, text)
    return cfg_dict
