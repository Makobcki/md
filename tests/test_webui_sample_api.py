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


def test_webui_sample_args_emit_direct_api_args(app_module: object, tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt.pt"
    control = tmp_path / "control.png"
    ckpt.write_bytes(b"x")
    control.write_bytes(b"x")

    args = app_module.SampleArgs(
        ckpt=str(ckpt),
        task="control",
        shift=3.0,
        **{"control-image": str(control), "fake-vae": True, "use-ema": False},
    )
    api_args = args.to_sample_options_args()

    assert api_args["control_image"] == str(control.resolve())
    assert api_args["fake_vae"] is True
    assert api_args["use_ema"] is False
    assert api_args["shift"] == 3.0
    assert "control-image" not in api_args


def test_webui_sample_manager_calls_sample_api_directly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from webui.backend import job_manager as jm

    calls = []

    def fake_run_sample(options, event_callback=None, quiet=False):
        calls.append(options)
        Path(options.out).parent.mkdir(parents=True, exist_ok=True)
        Path(options.out).write_text("fake", encoding="utf-8")
        if event_callback is not None:
            event_callback({"type": "metric", "step": 1, "max_steps": 1})
        return {"path": options.out}

    class _WS:
        def __init__(self) -> None:
            self.messages = []

        def set_loop(self, loop):
            return None

        def send_from_thread(self, run_id, channel, message):
            self.messages.append((run_id, channel, message))

    monkeypatch.setattr(jm, "run_sample", fake_run_sample)
    manager = jm.JobManager(tmp_path / "repo", _WS(), tmp_path / "runs")
    run = manager.start_sample(
        {
            "ckpt": "ckpt.pt",
            "prompt": "webui direct api",
            "steps": 1,
            "n": 1,
            "device": "cpu",
            "latent_only": True,
            "shift": 2.0,
        }
    )
    assert run.command[0] == "sample.api.run_sample"
    assert "sample.cli" not in " ".join(run.command)

    deadline = __import__("time").time() + 5
    while __import__("time").time() < deadline and manager.get_run(run.run_id).status in {"queued", "running"}:
        __import__("time").sleep(0.01)

    finished = manager.get_run(run.run_id)
    assert finished.status == "done"
    assert finished.exit_code == 0
    assert len(calls) == 1
    assert calls[0].shift == 2.0
    assert Path(finished.output_path).read_text(encoding="utf-8") == "fake"


def test_sample_args_endpoint_exposes_current_options(app_module: object) -> None:
    payload = app_module.get_sample_args()
    names = {item["name"] for item in payload["items"]}
    for expected in {
        "task",
        "sampler",
        "shift",
        "init-image",
        "strength",
        "mask",
        "control-image",
        "control-strength",
        "latent-only",
        "fake-vae",
        "use-ema",
    }:
        assert expected in names
