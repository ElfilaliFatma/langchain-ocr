"""Tolerant JSON extraction (shared pattern, kept local to stay self-contained)."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(raw: str) -> dict[str, Any] | None:
    if not raw or not raw.strip():
        return None
    candidates = [m.group(1) for m in _FENCE.finditer(raw)] + [raw]
    candidates.extend(balanced_objects(raw))
    for candidate in candidates:
        parsed = _try(candidate)
        if isinstance(parsed, dict):
            return parsed
    for candidate in candidates:
        parsed = _try(re.sub(r",\s*([}\]])", r"\1", candidate))
        if isinstance(parsed, dict):
            return parsed
    return None


def _try(text: str) -> Any:
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return None


def balanced_objects(text: str) -> list[str]:
    blocks: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                blocks.append(text[start : index + 1])
                start = -1
    return blocks
