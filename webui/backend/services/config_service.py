from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from config.formats.kdl_loader import KdlParseError, loads_kdl
from config.loader import load_train_config
from config.resolver import resolve_config
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


def _ensure_kdl_config_document(text: str) -> dict[str, Any]:
    stripped = text.lstrip()
    if not stripped:
        return {}
    if not stripped.startswith(("config ", "preset ")):
        raise ValueError("config editor accepts KDL documents only")
    return loads_kdl(text, source="<webui-config-editor>")


def _resolve_config_text(text: str, config_path: Path) -> dict[str, Any]:
    """Resolve editor text from a temporary sibling path for preset lookup."""

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=config_path.parent,
        encoding="utf-8",
        prefix=f".{config_path.stem}.",
        suffix=".kdl",
    ) as temp_file:
        temp_file.write(text)
        temp_path = Path(temp_file.name)
    try:
        return resolve_config(temp_path, expected_target="train").data
    finally:
        temp_path.unlink(missing_ok=True)


def _parse_config_mapping(text: str, config_path: Path | None = None) -> dict[str, Any]:
    data = _ensure_kdl_config_document(text)
    if not data:
        return {}
    if config_path is not None:
        return _resolve_config_text(text, config_path)
    return {key: value for key, value in data.items() if not key.startswith("__")}


def parse_config_text(text: str, config_path: Path | None = None) -> TrainConfig:
    data = _parse_config_mapping(text, config_path=config_path)
    return TrainConfig.from_dict(data)


def validate_config_text(text: str, config_path: Path | None = None) -> dict[str, Any]:
    cfg = parse_config_text(text, config_path=config_path)
    return asdict(cfg)


def write_config_text(repo_root: Path, text: str) -> dict[str, Any]:
    path = get_config_path(repo_root)
    try:
        cfg_dict = validate_config_text(text, config_path=path)
    except (KdlParseError, RuntimeError) as exc:
        raise ValueError(str(exc)) from exc
    atomic_write_text(path, text)
    return cfg_dict


def load_config_dict(repo_root: Path) -> dict[str, Any]:
    path = get_config_path(repo_root)
    return load_train_config(path).to_dict()
