from __future__ import annotations

from pathlib import Path
import importlib
from typing import Iterator

import pytest

from sample.api import SampleOptions


@pytest.fixture()
def app_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    cfg_src = Path(__file__).resolve().parents[1] / "config" / "train.yaml"
    cfg_dst = tmp_path / "train.yaml"
    out_dir = tmp_path / "external_out"
    content = "\n".join(
        f"out_dir: {out_dir}" if line.startswith("out_dir:") else line
        for line in cfg_src.read_text(encoding="utf-8").splitlines()
    )
    cfg_dst.write_text(content, encoding="utf-8")
    monkeypatch.setenv("WEBUI_CONFIG_PATH", str(cfg_dst))
    monkeypatch.setenv("WEBUI_RUNS_DIR", str(tmp_path / "webui_runs"))
    monkeypatch.setenv("WEBUI_ALLOWED_PATHS", str(tmp_path))
    import webui.backend.app as app_module
    importlib.reload(app_module)
    yield app_module


def test_sample_options_validate_txt2img() -> None:
    SampleOptions(ckpt="ckpt.pt", out="out.png", task="txt2img").validate()


def test_sample_options_validate_img2img_and_inpaint_require_inputs() -> None:
    with pytest.raises(RuntimeError, match="requires --init-image"):
        SampleOptions(ckpt="ckpt.pt", out="out.png", task="img2img").validate()
    with pytest.raises(RuntimeError, match="requires --mask"):
        SampleOptions(ckpt="ckpt.pt", out="out.png", task="inpaint", init_image="input.png").validate()
    SampleOptions(ckpt="ckpt.pt", out="out.png", task="inpaint", init_image="input.png", mask="mask.png").validate()


def test_sample_options_validate_control_requires_control_image() -> None:
    with pytest.raises(RuntimeError, match="requires --control-image"):
        SampleOptions(ckpt="ckpt.pt", out="out.png", task="control").validate()
    SampleOptions(ckpt="ckpt.pt", out="out.png", task="control", control_image="control.png").validate()


def test_webui_sample_args_emit_cli_aliases(app_module: object, tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.pt"
    init = tmp_path / "input.png"
    mask = tmp_path / "mask.png"
    control = tmp_path / "control.png"
    out = tmp_path / "out.png"
    for path in (ckpt, init, mask, control):
        path.write_bytes(b"x")

    args = app_module.SampleArgs(
        ckpt=str(ckpt),
        out=str(out),
        task="control",
        **{"control-image": str(control), "fake-vae": True},
    )
    cli = args.to_cli_args()

    assert cli["control-image"] == str(control.resolve())
    assert cli["fake-vae"] is True
    assert "use-ema" not in cli
