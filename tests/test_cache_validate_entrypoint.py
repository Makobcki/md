from __future__ import annotations

import pytest

from scripts.validate_cache import main


def test_validate_cache_help_opens(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "Validate text/latent training cache" in capsys.readouterr().out


def test_pyproject_exposes_cache_validate_entrypoint() -> None:
    text = open("pyproject.toml", encoding="utf-8").read()
    assert 'md-cache-validate = "scripts.validate_cache:main"' in text
    assert 'md-eval = "train.eval_cli:main"' in text
