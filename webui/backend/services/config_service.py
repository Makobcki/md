from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from config.formats.kdl_loader import KdlParseError, loads_kdl
from config.loader import load_train_config
from config.train import TrainConfig

from .atomic import atomic_write_text


def get_config_path(repo_root: Path) -> Path:
    override = os.environ.get("WEBUI_CONFIG_PATH")
    if override:
        return Path(override)
    return repo_root / "configs" / "train.kdl"


def read_config_text(repo_root: Path) -> str:
    path = get_config_path(repo_root)
    return path.read_text(encoding="utf-8")


def _parse_config_mapping(text: str) -> dict[str, Any]:
    stripped = text.lstrip()
    if not stripped:
        return {}
    if not stripped.startswith(("config ", "preset ")):
        raise ValueError("config editor accepts KDL documents only")
    data = loads_kdl(text, source="<webui-config-editor>")
    return {key: value for key, value in data.items() if not key.startswith("__")}


def parse_config_text(text: str) -> TrainConfig:
    data = _parse_config_mapping(text)
    return TrainConfig.from_dict(data)


def validate_config_text(text: str) -> dict[str, Any]:
    cfg = parse_config_text(text)
    return asdict(cfg)


def write_config_text(repo_root: Path, text: str) -> dict[str, Any]:
    try:
        cfg_dict = validate_config_text(text)
    except KdlParseError as exc:
        raise ValueError(str(exc)) from exc
    path = get_config_path(repo_root)
    atomic_write_text(path, text)
    return cfg_dict


def load_config_dict(repo_root: Path) -> dict[str, Any]:
    path = get_config_path(repo_root)
    return load_train_config(path).to_dict()
