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


def test_sample_metadata_helpers_write_json(tmp_path: Path) -> None:
    from argparse import Namespace

    source = _sample_cli_source()
    tree = ast.parse(source)
    helper_names = {"_metadata_sidecar_path", "_write_sample_metadata", "_sample_metadata"}
    helper_defs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in helper_names]
    module = ast.Module(body=helper_defs, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Path": Path, "json": __import__("json")}
    exec(compile(module, "sample_cli_helpers", "exec"), namespace)

    out = tmp_path / "sample.png"
    sidecar = namespace["_metadata_sidecar_path"](out)
    assert sidecar == tmp_path / "sample.json"

    args = Namespace(
        ckpt="./runs/mmdit_smoke/ckpt_final.pt",
        prompt="1girl, simple background",
        neg_prompt="",
        neg="low quality",
        steps=2,
        cfg=1,
        n=1,
    )
    built = Namespace(ckpt={"architecture": "mmdit_rf"}, cfg={})
    metadata = namespace["_sample_metadata"](args, built, sampler="flow_heun", seed=42)
    written = namespace["_write_sample_metadata"](out, metadata)

    assert written == sidecar
    payload = written.read_text(encoding="utf-8")
    assert payload.startswith("{\n")
    assert '"architecture": "mmdit_rf"' in payload
    assert '"sampler": "flow_heun"' in payload
