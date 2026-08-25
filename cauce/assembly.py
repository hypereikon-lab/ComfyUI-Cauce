"""Deterministic decoded-media assembly operations."""

from __future__ import annotations

from typing import Any


def accept_decoded_range(images: Any, start_frame: int, frame_count: int):
    """Return an exact visible-frame slice from a decoded batch."""

    start = int(start_frame)
    count = int(frame_count)
    end = start + count
    total = int(images.shape[0])
    if start < 0 or count < 1:
        raise ValueError("decoded range requires start_frame >= 0 and frame_count >= 1")
    if end > total:
        raise ValueError(f"decoded batch has {total} frames but the range ends at {end}")
    return images[start:end], count
