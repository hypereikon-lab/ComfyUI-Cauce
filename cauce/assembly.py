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


def restore_decoded_anchors(
    generated_images: Any,
    source_images: Any,
    factor: int = 2,
):
    """Restore every source frame onto an exact dense decoded-frame lattice.

    The generative pass owns only the frames between anchors.  This operation
    deliberately happens after H3 decoding because the H3 visual VAE cannot
    represent an alternating decoded-frame mask independently at its temporal
    token resolution.
    """

    multiplier = int(factor)
    if multiplier < 2 or multiplier > 4:
        raise ValueError("factor must be an integer from 2 through 4")

    source_count = int(source_images.shape[0])
    generated_count = int(generated_images.shape[0])
    if source_count < 2:
        raise ValueError("at least two source frames are required")
    if tuple(source_images.shape[1:]) != tuple(generated_images.shape[1:]):
        raise ValueError("source and generated frames must have matching geometry")

    delivery_count = (source_count - 1) * multiplier + 1
    if generated_count < delivery_count:
        raise ValueError(
            f"generated batch has {generated_count} frames but exact delivery "
            f"requires {delivery_count}"
        )

    accepted = generated_images[:delivery_count]
    restored = accepted.clone() if hasattr(accepted, "clone") else accepted.copy()
    restored[0:delivery_count:multiplier] = source_images
    generated_between_count = delivery_count - source_count
    return restored, delivery_count, source_count, generated_between_count
