from __future__ import annotations

import json
import sys
from pathlib import Path

from train.eval import DEFAULT_EVAL_PROMPT_SETS, load_eval_prompt_bank, load_eval_prompt_set
from train.eval_cli import main as eval_cli_main


def test_default_eval_prompt_bank_files_exist_and_ignore_empty_lines() -> None:
    bank = load_eval_prompt_bank(count_per_set=2)
    assert set(DEFAULT_EVAL_PROMPT_SETS) <= set(bank)
    for name, prompts in bank.items():
        assert prompts, name
        assert all(prompt.strip() for prompt in prompts)


def test_eval_prompt_set_loader_rejects_path_escape() -> None:
    import pytest

    with pytest.raises(RuntimeError, match="Invalid"):
        load_eval_prompt_set("../core")


def test_eval_cli_writes_prompt_set_metadata(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "prompts"
    root.mkdir()
    (root / "core.txt").write_text("first\n\nsecond\n", encoding="utf-8")
    (root / "style.txt").write_text("stylized\n", encoding="utf-8")
    out = tmp_path / "metadata.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.eval_cli",
            "--prompt-root",
            str(root),
            "--prompt-set",
            "core",
            "--prompt-set",
            "style",
            "--count-per-set",
            "1",
            "--metadata-out",
            str(out),
        ],
    )
    eval_cli_main()
    metadata = json.loads(out.read_text(encoding="utf-8"))
    assert [item["name"] for item in metadata["prompt_sets"]] == ["core", "style"]
    assert metadata["prompt_sets"][0]["prompts"] == ["first"]
    assert metadata["prompt_sets"][1]["count"] == 1
