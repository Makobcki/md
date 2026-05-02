from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from train.eval import _resolve_eval_prompts


def test_image_only_eval_uses_empty_prompts_without_prompt_file(tmp_path: Path) -> None:
    missing_prompts = tmp_path / "missing.txt"

    prompts = _resolve_eval_prompts(str(missing_prompts), count=5, use_text_conditioning=False)

    assert prompts == ["", "", "", "", ""]


def test_text_conditioned_eval_still_requires_prompt_file(tmp_path: Path) -> None:
    missing_prompts = tmp_path / "missing.txt"

    with pytest.raises(RuntimeError, match="Eval prompts file not found"):
        _resolve_eval_prompts(str(missing_prompts), count=5, use_text_conditioning=True)
