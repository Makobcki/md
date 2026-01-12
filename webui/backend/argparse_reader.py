from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List


def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    return None


def parse_argparse_args(sample_path: Path) -> List[Dict[str, Any]]:
    tree = ast.parse(sample_path.read_text(encoding="utf-8"))
    args: List[Dict[str, Any]] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> Any:
            if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                flags = []
                if node.args:
                    for arg in node.args:
                        val = _literal(arg)
                        if isinstance(val, str):
                            flags.append(val)
                if not flags:
                    return

                spec: Dict[str, Any] = {
                    "flags": flags,
                    "name": next((f.lstrip("-") for f in flags if f.startswith("--")), flags[0].lstrip("-")),
                    "type": "str",
                    "default": None,
                    "required": False,
                    "help": "",
                    "choices": None,
                }

                for kw in node.keywords:
                    if kw.arg == "type":
                        if isinstance(kw.value, ast.Name):
                            spec["type"] = kw.value.id
                    elif kw.arg == "default":
                        spec["default"] = _literal(kw.value)
                    elif kw.arg == "required":
                        spec["required"] = bool(_literal(kw.value))
                    elif kw.arg == "help":
                        spec["help"] = _literal(kw.value) or ""
                    elif kw.arg == "choices":
                        if isinstance(kw.value, (ast.List, ast.Tuple)):
                            spec["choices"] = [_literal(el) for el in kw.value.elts]

                args.append(spec)
            self.generic_visit(node)

    Visitor().visit(tree)
    return args
