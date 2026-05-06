from __future__ import annotations

import argparse
import py_compile
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

DEFAULT_PATHS: tuple[str, ...] = (
    "config",
    "control",
    "data_loader",
    "diffusion",
    "main.py",
    "model",
    "sample",
    "samplers",
    "scripts",
    "tests",
    "tg_bot",
    "train",
    "webui",
)
EXCLUDED_PATH_PARTS: frozenset[str] = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "data/dataset",
        "data_loader/raw",
        "runs",
        "samples",
        "vae_sd_mse",
        "webui/frontend/dist",
        "webui/frontend/node_modules",
        "webui_runs",
    }
)


def _project_root() -> Path:
    """Return the repository root for the installed or source-tree package."""
    return Path(__file__).resolve().parents[1]


def _is_relative_to(path: Path, parent: Path) -> bool:
    """Return whether `path` is located under `parent`.

    Args:
        path: Candidate child path.
        parent: Candidate parent path.

    Returns:
        True when `path` can be expressed relative to `parent`.

    """
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _display_path(path: Path, root: Path) -> str:
    """Format paths consistently for CLI output and subprocess arguments."""
    if _is_relative_to(path, root):
        return path.relative_to(root).as_posix()
    return path.as_posix()


def _is_excluded(path: Path, root: Path) -> bool:
    """Return whether `path` should be skipped by the project linter."""
    display_path = _display_path(path, root)
    parts = set(path.parts)
    return any(
        excluded in parts or display_path == excluded or display_path.startswith(f"{excluded}/")
        for excluded in EXCLUDED_PATH_PARTS
    )


def _existing_paths(raw_paths: Sequence[str], root: Path) -> list[Path]:
    """Resolve CLI paths and keep only existing paths.

    Args:
        raw_paths: User-provided paths relative to the project root or absolute paths.
        root: Project root.

    Returns:
        Existing resolved paths in input order.

    """
    paths: list[Path] = []
    for raw_path in raw_paths:
        path = Path(raw_path)
        resolved = path if path.is_absolute() else root / path
        if resolved.exists():
            paths.append(resolved.resolve())
        else:
            print(
                f"warning: lint path does not exist and will be skipped: {raw_path}",
                file=sys.stderr,
            )
    return paths


def _iter_python_files(paths: Iterable[Path], root: Path) -> list[Path]:
    """Collect Python source files from files and directories.

    Args:
        paths: Existing files or directories to scan.
        root: Project root.

    Returns:
        Sorted unique Python files, excluding generated and heavy directories.

    """
    files: set[Path] = set()
    for path in paths:
        if _is_excluded(path, root):
            continue
        if path.is_file() and path.suffix == ".py":
            files.add(path)
        elif path.is_dir():
            for candidate in path.rglob("*.py"):
                if not _is_excluded(candidate, root):
                    files.add(candidate)
    return sorted(files)


def _run_py_compile(paths: Sequence[Path], root: Path) -> int:
    """Compile Python files to catch syntax errors before heavier lint checks."""
    python_files = _iter_python_files(paths, root)
    if not python_files:
        print("py_compile: no Python files found")
        return 0

    print(f"py_compile: checking {len(python_files)} files")
    failures: list[Path] = []
    for path in python_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(path)
            print(f"{_display_path(path, root)}: {exc.msg}", file=sys.stderr)

    return 1 if failures else 0


def _normalize_source_text(text: str) -> str:
    """Normalize source text using checks implemented by this script.

    Args:
        text: Original UTF-8 source text.

    Returns:
        Text with LF endings, no trailing horizontal whitespace, and a final newline.

    """
    if not text:
        return text
    lines = text.splitlines()
    normalized = "\n".join(line.rstrip(" \t") for line in lines)
    return f"{normalized}\n"


def _source_hygiene_issues(path: Path, text: str, root: Path) -> list[str]:
    """Return source hygiene issues for one Python file."""
    display_path = _display_path(path, root)
    issues: list[str] = []

    for line_number, raw_line in enumerate(text.splitlines(keepends=True), start=1):
        line = raw_line.rstrip("\r\n")
        if line.endswith((" ", "\t")):
            issues.append(f"{display_path}:{line_number}: trailing whitespace")

    if text and not text.endswith("\n"):
        issues.append(f"{display_path}: missing final newline")
    if "\r\n" in text or "\r" in text.replace("\r\n", ""):
        issues.append(f"{display_path}: non-LF line endings")

    return issues


def _run_source_hygiene(paths: Sequence[Path], root: Path, *, fix: bool) -> int:
    """Check project source hygiene without external linter dependencies.

    Args:
        paths: Existing files or directories to check.
        root: Project root.
        fix: Whether this script may apply supported source hygiene fixes.

    Returns:
        Process return code.

    """
    python_files = _iter_python_files(paths, root)
    if not python_files:
        print("source_hygiene: no Python files found")
        return 0

    print(f"source_hygiene: checking {len(python_files)} files")
    has_failures = False
    for path in python_files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            has_failures = True
            print(f"{_display_path(path, root)}: invalid UTF-8: {exc}", file=sys.stderr)
            continue

        normalized = _normalize_source_text(text)
        if normalized == text:
            continue

        if fix:
            path.write_text(normalized, encoding="utf-8", newline="\n")
            print(f"fixed: {_display_path(path, root)}")
            continue

        has_failures = True
        for issue in _source_hygiene_issues(path, text, root):
            print(issue, file=sys.stderr)

    return 1 if has_failures else 0


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run project syntax and source hygiene checks without external linters."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=list(DEFAULT_PATHS),
        help="Files or directories to lint. Defaults to project Python packages and tests.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply supported source hygiene fixes instead of only checking them.",
    )
    parser.add_argument(
        "--no-py-compile",
        action="store_true",
        help="Skip Python bytecode compilation syntax checks.",
    )
    parser.add_argument(
        "--no-style",
        action="store_true",
        help="Skip source hygiene checks.",
    )
    parser.add_argument(
        "--no-ruff",
        action="store_true",
        dest="no_style",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-ruff-if-missing",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the project linter.

    Args:
        argv: Optional CLI arguments without the program name.

    Returns:
        Process return code.

    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    root = _project_root()
    paths = _existing_paths(args.paths, root)
    if not paths:
        print("error: no existing lint paths were provided", file=sys.stderr)
        return 2

    status = 0
    if not args.no_py_compile:
        status = _run_py_compile(paths, root) or status
    if not args.no_style:
        status = _run_source_hygiene(paths, root, fix=args.fix) or status
    return status


if __name__ == "__main__":
    raise SystemExit(main())
