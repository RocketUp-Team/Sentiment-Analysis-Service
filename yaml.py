from __future__ import annotations

import ast
import json
from typing import Any, Iterable, List, Tuple


def dump(data: Any, stream=None, **_: Any) -> str | None:
    text = json.dumps(data, indent=2, ensure_ascii=False)
    if stream is None:
        return text
    stream.write(text)
    return None


def safe_load(stream: Any) -> Any:
    text = stream.read() if hasattr(stream, "read") else str(stream)
    stripped = text.strip()
    if not stripped:
        return None

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        lines = _preprocess_yaml_lines(text.splitlines())
        value, index = _parse_block(lines, 0, 0)
        while index < len(lines) and not lines[index].strip():
            index += 1
        return value


def _preprocess_yaml_lines(raw_lines: Iterable[str]) -> List[str]:
    lines: List[str] = []
    for line in raw_lines:
        stripped = line.rstrip()
        if not stripped:
            continue
        if stripped.lstrip().startswith("#"):
            continue
        lines.append(stripped)
    return lines


def _parse_block(lines: List[str], index: int, indent: int) -> Tuple[Any, int]:
    items: Any = {}
    is_list = False

    while index < len(lines):
        line = lines[index]
        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent < indent:
            break
        if current_indent > indent:
            index += 1
            continue

        content = line.strip()
        if content.startswith("- "):
            if not is_list:
                items = []
                is_list = True
            value_text = content[2:].strip()
            if not value_text:
                value, index = _parse_block(lines, index + 1, indent + 2)
                items.append(value)
                continue
            items.append(_parse_scalar(value_text))
            index += 1
            continue

        key, sep, remainder = content.partition(":")
        if not sep:
            index += 1
            continue

        key = _parse_scalar(key.strip())
        remainder = remainder.strip()
        if remainder:
            items[key] = _parse_scalar(remainder)
            index += 1
            continue

        value, index = _parse_block(lines, index + 1, indent + 2)
        items[key] = value

    return items, index


def _parse_scalar(text: str) -> Any:
    if text in {"true", "True"}:
        return True
    if text in {"false", "False"}:
        return False
    if text in {"null", "Null", "none", "None"}:
        return None

    if text.startswith("[") and text.endswith("]"):
        try:
            return ast.literal_eval(
                text.replace("true", "True")
                .replace("false", "False")
                .replace("null", "None")
            )
        except (SyntaxError, ValueError):
            pass

    if (
        (text.startswith('"') and text.endswith('"'))
        or (text.startswith("'") and text.endswith("'"))
    ):
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text[1:-1]

    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text
