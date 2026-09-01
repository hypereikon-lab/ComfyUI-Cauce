"""Shared, deliberately thin helpers for ComfyUI node bindings."""

from __future__ import annotations

import json
from typing import Any

from ..cauce.comfy_compat import (
    conditioning_set_values,
    make_nested_tensor,
    require_arbitrary_h3_guides,
)
from ..cauce.comfy_compat.conditioning import existing_h3_keyframes


def json_report(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


__all__ = [
    "conditioning_set_values",
    "existing_h3_keyframes",
    "json_report",
    "make_nested_tensor",
    "require_arbitrary_h3_guides",
]
