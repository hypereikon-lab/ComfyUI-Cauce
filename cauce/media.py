"""Small, model-agnostic operations over opaque image batches."""

from __future__ import annotations

from typing import Any


def select_image_frame(images: Any, frame_index: int = -1):
    """Return one frame while preserving ComfyUI's IMAGE batch dimension."""

    frame_count = int(images.shape[0])
    if frame_count < 1:
        raise ValueError("cannot select a frame from an empty image batch")
    requested = int(frame_index)
    resolved = requested if requested >= 0 else frame_count + requested
    if resolved < 0 or resolved >= frame_count:
        raise ValueError(
            f"frame_index {requested} is outside an image batch of {frame_count} frames"
        )
    return images[resolved : resolved + 1], resolved
