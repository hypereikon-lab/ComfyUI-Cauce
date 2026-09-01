"""Adapter for immutable ComfyUI conditioning metadata updates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def existing_h3_keyframes(positive: Any) -> list[dict[str, Any]]:
    if not isinstance(positive, list) or not positive:
        raise TypeError("positive must be a non-empty ComfyUI CONDITIONING list")
    first = positive[0]
    if not isinstance(first, (list, tuple)) or len(first) < 2:
        raise TypeError("positive entries must contain a tensor and metadata mapping")
    metadata = first[1]
    if not isinstance(metadata, Mapping):
        raise TypeError("positive conditioning metadata must be a mapping")
    keyframes: Sequence[Any] = metadata.get("minimax_keyframes", [])
    if not isinstance(keyframes, (list, tuple)):
        raise TypeError("positive minimax_keyframes metadata must be a list or tuple")
    if not all(isinstance(keyframe, Mapping) for keyframe in keyframes):
        raise TypeError("every positive minimax_keyframes entry must be a mapping")
    return [dict(keyframe) for keyframe in keyframes]


def conditioning_set_values(conditioning: Any, values: Mapping[str, Any]) -> Any:
    try:
        import node_helpers
    except ImportError as exc:  # pragma: no cover - requires ComfyUI
        raise RuntimeError("conditioning updates require ComfyUI node_helpers") from exc
    setter = getattr(node_helpers, "conditioning_set_values", None)
    if not callable(setter):
        raise RuntimeError("this ComfyUI build lacks conditioning_set_values")
    return setter(conditioning, dict(values))
