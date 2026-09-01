"""Adapter for ComfyUI's nested audiovisual tensor carrier."""

from __future__ import annotations

from typing import Any


def make_nested_tensor(streams: tuple[Any, Any]) -> Any:
    try:
        from comfy.nested_tensor import NestedTensor
    except ImportError as exc:  # pragma: no cover - requires ComfyUI
        raise RuntimeError("H3 audiovisual carriers require ComfyUI NestedTensor") from exc
    return NestedTensor(streams)
