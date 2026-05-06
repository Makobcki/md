from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from scripts import lint


def test_lint_help_opens(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        lint.main(["--help"])

    assert exc.value.code == 0
    assert "Run project syntax checks" in capsys.readouterr().out


def test_lint_runs_py_compile_and_ruff_checks(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    commands: list[list[str]] = []

    def fake_which(name: str) -> str | None:
        assert name == "ruff"
        return "ruff"

    def fake_run(
        command: Sequence[str],
        cwd: Path,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == lint._project_root()
        assert check is False
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(lint.shutil, "which", fake_which)
    monkeypatch.setattr(lint.subprocess, "run", fake_run)

    assert lint.main(["scripts/lint.py"]) == 0

    assert commands == [
        ["ruff", "check", "scripts/lint.py"],
        ["ruff", "format", "--check", "scripts/lint.py"],
    ]
    assert "py_compile: checking 1 files" in capsys.readouterr().out


def test_lint_allows_missing_ruff_to_be_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lint.shutil, "which", lambda name: None)

    assert lint.main(["--no-py-compile", "--skip-ruff-if-missing", "scripts/lint.py"]) == 0


def test_pyproject_exposes_lint_entrypoint_and_ruff_config() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'md-lint = "scripts.lint:main"' in text
    assert '"ruff>=0.8.0"' in text
    assert "[tool.ruff]" in text
