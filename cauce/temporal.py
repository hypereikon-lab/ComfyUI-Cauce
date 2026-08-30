"""Deterministic planning for frame-rate conversion and H3 temporal geometry."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from .contracts import content_hash
from .timebase import (
    H3_FPS,
    as_fraction,
    ceil_h3_frame_count,
    fraction_payload,
    h3_visual_latent_frames,
    round_fraction,
    visual_token_spans,
)


def _positive_int(value: int, label: str, *, minimum: int = 1) -> int:
    resolved = int(value)
    if resolved < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return resolved


def _fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "fraction": fraction_payload(value),
        "decimal": float(value),
    }


def plan_frame_interpolation(
    source_frame_count: int,
    multiplier: int,
    source_fps: Fraction | int | float | str = H3_FPS,
) -> dict[str, Any]:
    """Plan endpoint-preserving decoded-frame interpolation.

    A multiplier ``m`` inserts ``m - 1`` frames between every adjacent source
    pair. The output therefore contains ``(N - 1) * m + 1`` samples. This is
    intentionally not reported as ``N * m``: preserving both endpoints makes
    the exact sample-grid result one frame shorter than that common shorthand.
    """

    frames = _positive_int(source_frame_count, "source_frame_count", minimum=2)
    factor = _positive_int(multiplier, "multiplier")
    fps = as_fraction(source_fps)
    if fps <= 0:
        raise ValueError("source_fps must be positive")

    pair_count = frames - 1
    target_frame_count = pair_count * factor + 1
    target_fps = fps * factor
    source_anchor_indices = [index * factor for index in range(frames)]
    inserted_frame_count = pair_count * (factor - 1)
    sample_span_duration = Fraction(pair_count, 1) / fps
    source_container_duration = Fraction(frames, 1) / fps
    target_container_duration = Fraction(target_frame_count, 1) / target_fps

    plan: dict[str, Any] = {
        "schema": "cauce.frame-interpolation-plan/1",
        "method_class": "decoded-frame-interpolation",
        "source_frame_count": frames,
        "source_fps": _fraction_record(fps),
        "pair_count": pair_count,
        "multiplier": factor,
        "target_frame_count": target_frame_count,
        "target_fps": _fraction_record(target_fps),
        "inserted_frame_count": inserted_frame_count,
        "source_anchor_output_indices": source_anchor_indices,
        "sample_span_duration": _fraction_record(sample_span_duration),
        "source_container_duration": _fraction_record(source_container_duration),
        "target_container_duration": _fraction_record(target_container_duration),
        "endpoint_preserving": True,
        "changes_semantic_motion": False,
        "notes": [
            "Every source sample remains at output index source_index * multiplier.",
            "The exact output count is (N - 1) * multiplier + 1, not N * multiplier.",
            "Container-duration metadata differs by one sample interval; the first-to-last sample span is invariant.",
        ],
    }
    plan["plan_hash"] = content_hash(plan)
    return plan


def analyze_h3_interleave_projection(
    source_frame_count: int,
    multiplier: int,
) -> dict[str, Any]:
    """Project an alternating known/missing decoded mask onto H3 tokens.

    The official H3 conditioning path reduces every temporal token span using
    maximum mask strength. A token containing even one requested/generated
    decoded frame is therefore generated as a whole. This report makes mixed
    known/missing tokens visible instead of pretending H3 has one latent token
    per output frame.
    """

    interpolation = plan_frame_interpolation(source_frame_count, multiplier, H3_FPS)
    raw_target = int(interpolation["target_frame_count"])
    resolved_target = ceil_h3_frame_count(raw_target)
    known = set(interpolation["source_anchor_output_indices"])
    token_spans = visual_token_spans(h3_visual_latent_frames(resolved_target))

    records: list[dict[str, Any]] = []
    preserve_only = 0
    generate_only = 0
    mixed = 0
    for token_index, (start, end) in enumerate(token_spans):
        known_indices = [index for index in range(start, end) if index in known]
        missing_indices = [index for index in range(start, end) if index not in known]
        if not missing_indices:
            classification = "preserve-only"
            projected_mask = 0.0
            preserve_only += 1
        elif not known_indices:
            classification = "generate-only"
            projected_mask = 1.0
            generate_only += 1
        else:
            classification = "mixed-known-and-missing"
            projected_mask = 1.0
            mixed += 1
        records.append(
            {
                "token_index": token_index,
                "decoded_span": [start, end],
                "known_frame_indices": known_indices,
                "missing_frame_indices": missing_indices,
                "classification": classification,
                "projected_mask": projected_mask,
            }
        )

    exact = mixed == 0
    report: dict[str, Any] = {
        "schema": "cauce.h3-interleave-projection/1",
        "source_frame_count": int(source_frame_count),
        "multiplier": int(multiplier),
        "raw_interpolated_frame_count": raw_target,
        "resolved_h3_frame_count": resolved_target,
        "h3_padding_frame_count": resolved_target - raw_target,
        "video_token_count": len(token_spans),
        "known_decoded_frame_count": len(known),
        "requested_generated_frame_count": resolved_target - len(known),
        "token_counts": {
            "preserve_only": preserve_only,
            "mixed_known_and_missing": mixed,
            "generate_only": generate_only,
        },
        "exact_known_frame_preservation_possible": exact,
        "official_projection_rule": "temporal maximum over each H3 decoded-frame token span",
        "tokens": records,
        "conclusion": (
            "The proposed interleave mask aligns with H3 temporal tokens."
            if exact
            else "The proposed interleave mask mixes preserved and generated decoded frames inside H3 temporal tokens; those mixed tokens are regenerated as a unit."
        ),
    }
    report["report_hash"] = content_hash(report)
    return report


def plan_h3_guide_retime(
    source_frame_count: int,
    duration_scale: Fraction | int | float | str,
    guide_stride_source_frames: int = 24,
    source_fps: Fraction | int | float | str = H3_FPS,
) -> dict[str, Any]:
    """Plan sparse still-guide placement for creative H3 retiming.

    This operation keeps H3 at its native 24 fps. It changes the requested
    duration and maps selected source frames to arbitrary-frame official H3
    guides. The legal ``17k + 5`` lattice can lengthen the requested span; the
    first and final guides are therefore aligned to the resolved endpoints.
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
    requested_target_frames = round_fraction(requested_sample_span * H3_FPS) + 1
    requested_target_frames = max(5, requested_target_frames)
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
