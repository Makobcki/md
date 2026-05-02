from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sample_cli_source() -> str:
    return (ROOT / "sample" / "cli.py").read_text(encoding="utf-8")


def test_sample_cli_defines_metadata_sidecar_helpers() -> None:
    tree = ast.parse(_sample_cli_source())
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

    assert "_metadata_sidecar_path" in functions
    assert "_write_sample_metadata" in functions
    assert "_sample_metadata" in functions


def test_sample_cli_writes_metadata_next_to_image() -> None:
    source = _sample_cli_source()

    assert 'Path(image_path).with_suffix(".json")' in source
    assert '"architecture": architecture' in source
    assert '"prompt": str(args.prompt)' in source
    assert '"sampler": str(sampler)' in source
    assert '"steps": int(args.steps)' in source
    assert '"cfg": float(args.cfg)' in source
    assert '"seed": int(seed)' in source
    assert '"n": int(args.n)' in source
    assert source.count("_write_sample_metadata(out, _sample_metadata") == 2
