from __future__ import annotations

from pathlib import Path

import pytest

from scripts import lint


def test_lint_help_opens(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        lint.main(["--help"])

    assert exc.value.code == 0
    assert "Run project syntax and source hygiene checks" in capsys.readouterr().out


def test_lint_runs_py_compile_and_source_hygiene(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert lint.main(["scripts/lint.py"]) == 0

    output = capsys.readouterr().out
    assert "py_compile: checking 1 files" in output
    assert "source_hygiene: checking 1 files" in output


def test_lint_fails_on_source_hygiene_issues(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_path = tmp_path / "bad.py"
    source_path.write_text("value = 1  ", encoding="utf-8")

    assert lint.main(["--no-py-compile", str(source_path)]) == 1

    errors = capsys.readouterr().err
    assert "trailing whitespace" in errors
    assert "missing final newline" in errors


def test_lint_fix_applies_source_hygiene_fixes(tmp_path: Path) -> None:
    source_path = tmp_path / "bad.py"
    source_path.write_bytes(b"value = 1  \r\n")

    assert lint.main(["--no-py-compile", "--fix", str(source_path)]) == 0

    assert source_path.read_bytes() == b"value = 1\n"


def test_lint_keeps_legacy_ruff_flags_as_noops() -> None:
    assert lint.main(["--no-py-compile", "--skip-ruff-if-missing", "--no-ruff", "scripts/lint.py"]) == 0


def test_pyproject_exposes_lint_entrypoint_without_ruff_dependency() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'md-lint = "scripts.lint:main"' in text
    assert '"ruff>=0.8.0"' not in text
    assert "[tool.ruff]" in text
