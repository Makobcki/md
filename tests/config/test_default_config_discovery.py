from __future__ import annotations

from config.discovery import default_config_path


def test_default_config_discovery_by_target() -> None:
    assert default_config_path("train").as_posix().endswith("configs/train.kdl")
    assert default_config_path("sample").as_posix().endswith("configs/sample.kdl")
    assert default_config_path("webui").as_posix().endswith("configs/webui.kdl")
    assert default_config_path("eval").as_posix().endswith("configs/eval.kdl")
    assert default_config_path("prepare-latents").as_posix().endswith("configs/cache.kdl")
