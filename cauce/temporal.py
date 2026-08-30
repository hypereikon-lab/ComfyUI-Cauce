"""Deterministic planning for official H3 arbitrary-frame guide retiming."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from .contracts import content_hash
from .timebase import H3_FPS, as_fraction, ceil_h3_frame_count, fraction_payload, round_fraction


def _positive_int(value: int, label: str, *, minimum: int = 1) -> int:
    resolved = int(value)
    if resolved < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return resolved


def _fraction_record(value: Fraction) -> dict[str, Any]:
    return {"fraction": fraction_payload(value), "decimal": float(value)}


def plan_h3_guide_retime(
    source_frame_count: int,
    duration_scale: Fraction | int | float | str,
    guide_stride_source_frames: int = 24,
    source_fps: Fraction | int | float | str = H3_FPS,
) -> dict[str, Any]:
    """Map sparse source frames to official H3 arbitrary-frame guides.

    This is creative H3 regeneration at 24 fps. Native temporal densification
    of packed H3 state lives in :mod:`cauce.av_latent`; decoded-frame
    interpolation is intentionally absent.
    """

    frames = _positive_int(source_frame_count, "source_frame_count", minimum=2)
    stride = _positive_int(guide_stride_source_frames, "guide_stride_source_frames")
    fps = as_fraction(source_fps)
    scale = as_fraction(duration_scale)
    if fps <= 0:
        raise ValueError("source_fps must be positive")
    if scale <= 0:
        raise ValueError("duration_scale must be positive")

    requested_sample_span = Fraction(frames - 1, 1) / fps * scale
    requested_target_frames = max(5, round_fraction(requested_sample_span * H3_FPS) + 1)
    resolved_target_frames = ceil_h3_frame_count(requested_target_frames)

    source_indices = list(range(0, frames, stride))
    if source_indices[-1] != frames - 1:
        source_indices.append(frames - 1)
    target_last = resolved_target_frames - 1
    source_last = frames - 1
    anchors: list[dict[str, Any]] = []
    seen_targets: set[int] = set()
    for source_index in source_indices:
        target_index = round_fraction(Fraction(source_index * target_last, source_last))
        if target_index in seen_targets:
            continue
        seen_targets.add(target_index)
        anchors.append(
            {
                "source_frame_index": source_index,
                "target_frame_index": target_index,
                "target_time_seconds": float(Fraction(target_index, 1) / H3_FPS),
                "guide_type": "single-image",
            }
        )

    resolved_sample_span = Fraction(target_last, 1) / H3_FPS
    plan: dict[str, Any] = {
        "schema": "cauce.h3-guide-retime-plan/1",
        "method_class": "creative-h3-regeneration-with-sparse-still-guides",
        "source_frame_count": frames,
        "source_fps": _fraction_record(fps),
        "duration_scale": _fraction_record(scale),
        "guide_stride_source_frames": stride,
        "requested_target_frame_count": requested_target_frames,
        "resolved_h3_frame_count": resolved_target_frames,
        "h3_lattice_padding_frames": resolved_target_frames - requested_target_frames,
        "output_fps": _fraction_record(H3_FPS),
        "requested_sample_span_duration": _fraction_record(requested_sample_span),
        "resolved_sample_span_duration": _fraction_record(resolved_sample_span),
        "guide_count": len(anchors),
        "anchors": anchors,
        "endpoint_guides": True,
        "pixel_exact_interpolation": False,
        "changes_semantic_motion": True,
        "notes": [
            "H3 remains at 24 fps; this is creative retiming, not frame-rate conversion.",
            "Each anchor is an official arbitrary-frame single-image guide.",
            "The resolved output is snapped upward to the H3 17k+5 frame lattice.",
        ],
    }
    plan["plan_hash"] = content_hash(plan)
    return plan
