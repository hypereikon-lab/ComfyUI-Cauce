"""Visible planning for H3 target and temporal input preprocessing."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import (
    H3_GUIDE_PLAN_SCHEMA,
    H3_REFERENCE_PLAN_SCHEMA,
    H3_TARGET_PLAN_SCHEMA,
    content_hash,
)
from .h3 import validate_av_latent
from .timebase import (
    H3_TRAINED_MIN_FRAMES,
    H3Shape,
    ceil_h3_frame_count,
    floor_h3_frame_count,
)


H3_TRAINED_MAX_FRAMES = 362
H3_CANVAS_MULTIPLE = 32
H3_QWEN_VIDEO_STRIDE = 12  # 24 fps decoded media presented to Qwen at 2 fps.
H3_REFERENCE_DOCUMENTED_MIN_FRAMES = 2 * 24
H3_REFERENCE_DOCUMENTED_MAX_FRAMES = 15 * 24


def _with_hash(value: dict[str, Any], field: str) -> dict[str, Any]:
    value[field] = content_hash(value)
    return value


def _image_count(images: Any) -> int:
    shape = getattr(images, "shape", None)
    if shape is None or len(shape) < 1:
        raise TypeError("images must expose a batch dimension")
    count = int(shape[0])
    if count < 1:
        raise ValueError("images must contain at least one frame")
    return count


def resolve_h3_target_shape(
    requested_frames: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Expose the exact target geometry that official H3 nodes allocate."""

    requested = int(requested_frames)
    w = int(width)
    h = int(height)
    if requested < 1:
        raise ValueError("requested_frames must be positive")
    if w < H3_CANVAS_MULTIPLE or h < H3_CANVAS_MULTIPLE:
        raise ValueError("H3 target dimensions must be at least 32 pixels")
    if w % H3_CANVAS_MULTIPLE or h % H3_CANVAS_MULTIPLE:
        raise ValueError("H3 target dimensions must be multiples of 32")
    resolved = ceil_h3_frame_count(requested)
    shape = H3Shape.from_frames(resolved)
    payload: dict[str, Any] = {
        "schema": H3_TARGET_PLAN_SCHEMA,
        "requested_frames": requested,
        "resolved_frames": resolved,
        "added_frames": resolved - requested,
        "width": w,
        "height": h,
        "duration": shape.to_dict()["duration"],
        "duration_seconds": float(shape.duration),
        "video_tokens": shape.video_latent_frames,
        "audio_tokens": shape.audio_latent_frames,
        "inside_trained_range": H3_TRAINED_MIN_FRAMES <= resolved <= H3_TRAINED_MAX_FRAMES,
        "trained_range": [H3_TRAINED_MIN_FRAMES, H3_TRAINED_MAX_FRAMES],
    }
    return _with_hash(payload, "plan_hash")


def plan_h3_guide_clip(
    images: Any,
    target_av_latent: Mapping[str, Any],
    frame_idx: int,
    *,
    timeline_origin_frame: int = 0,
) -> dict[str, Any]:
    """Resolve official AddGuide clipping and index rules before VAE encoding."""

    source_frames = _image_count(images)
    _, _, target_frames = validate_av_latent(
        target_av_latent,
        timeline_origin_frame=int(timeline_origin_frame),
        name="target_av_latent",
    )
    accepted = 1 if source_frames < 5 else floor_h3_frame_count(source_frames)
    requested_index = int(frame_idx)
    resolved_index = (
        requested_index if requested_index >= 0 else target_frames + requested_index
    )
    end = resolved_index + accepted
    if resolved_index < 0 or end > target_frames:
        raise ValueError(
            f"a {accepted} frame guide at frame_idx {requested_index} does not fit "
            f"in the target's {target_frames} frames"
        )
    payload: dict[str, Any] = {
        "schema": H3_GUIDE_PLAN_SCHEMA,
        "source_frames": source_frames,
        "accepted_frames": accepted,
        "discarded_tail_frames": source_frames - accepted,
        "requested_frame_idx": requested_index,
        "resolved_frame_idx": resolved_index,
        "target_frames": target_frames,
        "timeline_origin_frame": int(timeline_origin_frame),
        "guide_range": [resolved_index, end],
        "single_image_mode": accepted == 1,
    }
    return _with_hash(payload, "plan_hash")


def prepare_h3_guide_clip(
    images: Any,
    target_av_latent: Mapping[str, Any],
    frame_idx: int,
    *,
    timeline_origin_frame: int = 0,
) -> tuple[Any, dict[str, Any]]:
    plan = plan_h3_guide_clip(
        images,
        target_av_latent,
        frame_idx,
        timeline_origin_frame=timeline_origin_frame,
    )
    return images[: int(plan["accepted_frames"])], plan


def plan_h3_reference_clip(
    images: Any,
    requested_target_frames: int,
) -> dict[str, Any]:
    """Expose official Ref2VA target clamping, lattice trim, and Qwen samples."""

    source_frames = _image_count(images)
    if source_frames < 5:
        raise ValueError("MiniMax H3 reference videos need at least 5 frames")
    requested_target = int(requested_target_frames)
    if requested_target < 1:
        raise ValueError("requested_target_frames must be positive")
    target_frames = ceil_h3_frame_count(requested_target)
    limited_frames = min(source_frames, target_frames)
    accepted = floor_h3_frame_count(limited_frames)
    sample_indices = list(range(0, accepted, H3_QWEN_VIDEO_STRIDE))
    payload: dict[str, Any] = {
        "schema": H3_REFERENCE_PLAN_SCHEMA,
        "source_frames": source_frames,
        "requested_target_frames": requested_target,
        "resolved_target_frames": target_frames,
        "accepted_frames": accepted,
        "discarded_tail_frames": source_frames - accepted,
        "duration_seconds": accepted / 24.0,
        "inside_documented_duration_range": (
            H3_REFERENCE_DOCUMENTED_MIN_FRAMES
            <= accepted
            <= H3_REFERENCE_DOCUMENTED_MAX_FRAMES
        ),
        "documented_duration_range_seconds": [2, 15],
        "documented_duration_range_frames": [
            H3_REFERENCE_DOCUMENTED_MIN_FRAMES,
            H3_REFERENCE_DOCUMENTED_MAX_FRAMES,
        ],
        "qwen_sample_indices": sample_indices,
        "qwen_timestamps_seconds": [index / 2.0 for index in range(len(sample_indices))],
        "qwen_sample_count": len(sample_indices),
    }
    return _with_hash(payload, "plan_hash")


def prepare_h3_reference_clip(
    images: Any,
    requested_target_frames: int,
) -> tuple[Any, dict[str, Any]]:
    plan = plan_h3_reference_clip(images, requested_target_frames)
    return images[: int(plan["accepted_frames"])], plan
