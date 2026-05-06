from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KdlNode:
    """Small KDL node representation for md-diffusion configuration files."""

    name: str
    args: list[Any] = field(default_factory=list)
    props: dict[str, Any] = field(default_factory=dict)
    children: list["KdlNode"] = field(default_factory=list)
    line: int = 1


class KdlParseError(RuntimeError):
    """Raised when a KDL configuration cannot be parsed."""


class _TokenStream:
    """Token stream for the subset of KDL used by project configs.

    The parser intentionally supports only the KDL surface needed for stable
    config files: nodes, quoted strings, bare identifiers, numeric/bool/null
    values, properties with ``=``, braces, and bracket lists.
    """

    def __init__(self, text: str, *, source: str) -> None:
        self._tokens = list(_tokenize(text, source=source))
        self._index = 0
        self.source = source

    def peek(self) -> tuple[str, Any, int] | None:
        if self._index >= len(self._tokens):
            return None
        return self._tokens[self._index]

    def pop(self) -> tuple[str, Any, int]:
        token = self.peek()
        if token is None:
            raise KdlParseError(f"{self.source}: unexpected end of file")
        self._index += 1
        return token

    def match(self, kind: str) -> bool:
        token = self.peek()
        if token is not None and token[0] == kind:
            self._index += 1
            return True
        return False

    def expect(self, kind: str) -> tuple[str, Any, int]:
        token = self.pop()
        if token[0] != kind:
            raise KdlParseError(
                f"{self.source}:{token[2]}: expected {kind}, got {token[0]}"
            )
        return token

    def skip_newlines(self) -> None:
        while self.match("NEWLINE"):
            pass


def _tokenize(text: str, *, source: str) -> list[tuple[str, Any, int]]:
    tokens: list[tuple[str, Any, int]] = []
    i = 0
    line = 1
    while i < len(text):
        ch = text[i]
        if ch in " \t\r":
            i += 1
            continue
        if ch == "\n" or ch == ";":
            tokens.append(("NEWLINE", None, line))
            line += 1 if ch == "\n" else 0
            i += 1
            continue
        if ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
            i += 2
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        if ch == "#":
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        if ch == "/" and i + 1 < len(text) and text[i + 1] == "*":
            i += 2
            while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                if text[i] == "\n":
                    line += 1
                i += 1
            if i + 1 >= len(text):
                raise KdlParseError(f"{source}:{line}: unterminated block comment")
            i += 2
            continue
        if ch in "{}[]=,":
            tokens.append((ch, ch, line))
            i += 1
            continue
        if ch == '"':
            value, i, line = _read_string(text, i, line, source=source)
            tokens.append(("VALUE", value, line))
            continue

        start = i
        while i < len(text) and text[i] not in " \t\r\n{}[]=,;":
            if text[i] == "/" and i + 1 < len(text) and text[i + 1] in {"/", "*"}:
                break
            i += 1
        if start == i:
            raise KdlParseError(f"{source}:{line}: unexpected character {ch!r}")
        raw = text[start:i]
        tokens.append(("IDENT", raw, line))
    return tokens


def _read_string(text: str, start: int, line: int, *, source: str) -> tuple[str, int, int]:
    assert text[start] == '"'
    i = start + 1
    chars: list[str] = []
    while i < len(text):
        ch = text[i]
        if ch == '"':
            return "".join(chars), i + 1, line
        if ch == "\\":
            i += 1
            if i >= len(text):
                raise KdlParseError(f"{source}:{line}: unterminated escape sequence")
            esc = text[i]
            mapping = {
                "n": "\n",
                "r": "\r",
                "t": "\t",
                "\\": "\\",
                '"': '"',
                "b": "\b",
                "f": "\f",
            }
            chars.append(mapping.get(esc, esc))
            i += 1
            continue
        if ch == "\n":
            line += 1
        chars.append(ch)
        i += 1
    raise KdlParseError(f"{source}:{line}: unterminated string")


def _parse_scalar(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    lowered = raw.lower()
    if lowered in {"true", "#true"}:
        return True
    if lowered in {"false", "#false"}:
        return False
    if lowered in {"null", "#null"}:
        return None
    try:
        if any(marker in raw for marker in (".", "e", "E")):
            return float(raw)
        return int(raw, 10)
    except ValueError:
        return raw


def _parse_value(stream: _TokenStream) -> Any:
    token = stream.pop()
    kind, value, line = token
    if kind == "VALUE":
        return value
    if kind == "IDENT":
        return _parse_scalar(value)
    if kind == "[":
        items: list[Any] = []
        while True:
            stream.skip_newlines()
            if stream.match("]"):
                return items
            items.append(_parse_value(stream))
            stream.skip_newlines()
            stream.match(",")
    raise KdlParseError(f"{stream.source}:{line}: expected value, got {kind}")


def _parse_node(stream: _TokenStream) -> KdlNode:
    name_token = stream.expect("IDENT")
    name = str(name_token[1])
    args: list[Any] = []
    props: dict[str, Any] = {}
    children: list[KdlNode] = []

    while True:
        token = stream.peek()
        if token is None or token[0] in {"NEWLINE", "}", "{"}:
            break
        if token[0] == ",":
            stream.pop()
            continue
        if token[0] == "IDENT":
            next_token = (
                stream._tokens[stream._index + 1]
                if stream._index + 1 < len(stream._tokens)
                else None
            )
            if next_token is not None and next_token[0] == "=":
                key = str(stream.pop()[1])
                stream.expect("=")
                props[key] = _parse_value(stream)
                continue
        args.append(_parse_value(stream))

    stream.skip_newlines()
    if stream.match("{"):
        while True:
            stream.skip_newlines()
            if stream.match("}"):
                break
            if stream.peek() is None:
                raise KdlParseError(f"{stream.source}:{name_token[2]}: unclosed node {name!r}")
            children.append(_parse_node(stream))
            stream.skip_newlines()
    return KdlNode(name=name, args=args, props=props, children=children, line=int(name_token[2]))


def parse_kdl(text: str, *, source: str = "<string>") -> list[KdlNode]:
    """Parse KDL text into nodes."""

    stream = _TokenStream(text, source=source)
    nodes: list[KdlNode] = []
    while True:
        stream.skip_newlines()
        if stream.peek() is None:
            return nodes
        nodes.append(_parse_node(stream))


def _node_value(node: KdlNode) -> Any:
    if node.children:
        data: dict[str, Any] = {}
        for key, value in node.props.items():
            data[f"@{key}"] = value
        for child in node.children:
            if child.name == "use":
                if not child.args:
                    raise KdlParseError("use node requires a path argument")
                data.setdefault("__uses__", []).append(str(child.args[0]))
                continue
            child_value = _node_value(child)
            if child.name in data:
                current = data[child.name]
                if isinstance(current, list):
                    current.append(child_value)
                else:
                    data[child.name] = [current, child_value]
            else:
                data[child.name] = child_value
        return data
    if node.props and not node.args:
        return dict(node.props)
    if node.props and node.args:
        data = dict(node.props)
        data["@args"] = list(node.args)
        return data
    if len(node.args) == 0:
        return True
    if len(node.args) == 1:
        return node.args[0]
    return list(node.args)


def _document_to_dict(nodes: list[KdlNode], *, source: str) -> dict[str, Any]:
    meaningful = [node for node in nodes if node.name in {"config", "preset"}]
    if len(meaningful) != 1:
        raise KdlParseError(
            f"{source}: expected exactly one top-level config or preset node"
        )
    root = meaningful[0]
    body = _node_value(root)
    if not isinstance(body, dict):
        body = {}
    meta = dict(root.props)
    for prop_name in meta:
        body.pop(f"@{prop_name}", None)
    uses = list(body.pop("__uses__", []))
    result: dict[str, Any] = {
        "__kind__": root.name,
        "__meta__": meta,
        "__source__": source,
    }
    if uses:
        result["__uses__"] = uses
    result.update(body)
    return result


def loads_kdl(text: str, *, source: str = "<string>") -> dict[str, Any]:
    """Load KDL text into a nested dictionary."""

    return _document_to_dict(parse_kdl(text, source=source), source=source)


def load_kdl(path: str | Path) -> dict[str, Any]:
    """Load a KDL config or preset file into a nested dictionary.

    The returned mapping contains ``__kind__`` (``config`` or ``preset``),
    ``__meta__`` with root node properties, and ``__uses__`` if present.
    """

    file_path = Path(path)
    return loads_kdl(file_path.read_text(encoding="utf-8"), source=str(file_path))
