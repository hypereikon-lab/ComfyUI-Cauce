"""H3 AV inspection, absolute window planning, and allocation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts import AV_WINDOW_LAYOUT_SCHEMA, content_hash
from ..h3 import validate_av_latent
from ..timebase import (
    h3_audio_token_boundary,
    h3_visual_latent_frames,
    is_h3_frame_count,
)
from .backend import (
    clone as _clone,
)
from .backend import (
    device_name as _device,
)
from .backend import (
    dtype_name as _dtype,
)
from .backend import (
    make_latent as _make_latent,
)
from .backend import (
    new_zeros as _new_zeros,
)
from .types import NestedFactory


def inspect_av_latent(
    latent: Mapping[str, Any],
    *,
    timeline_origin_frame: int = 0,
) -> dict[str, Any]:
    """Return a serializable report for one complete H3 latent or aligned window."""

    video, audio, frames = validate_av_latent(
        latent,
        timeline_origin_frame=timeline_origin_frame,
    )
    return {
        "schema": "cauce.h3-av-latent-report/1",
        "timeline_origin_frame": int(timeline_origin_frame),
        "timeline_end_frame": int(timeline_origin_frame) + frames,
        "frame_count": frames,
        "video_tokens": int(video.shape[2]),
        "audio_tokens": int(audio.shape[-1]),
        "video_shape": [int(item) for item in video.shape],
        "audio_shape": [int(item) for item in audio.shape],
        "video_dtype": _dtype(video),
        "audio_dtype": _dtype(audio),
        "video_device": _device(video),
        "audio_device": _device(audio),
    }


def plan_av_window(
    previous_av_latent: Mapping[str, Any],
    *,
    overlap_frames: int,
    extension_frames: int,
) -> dict[str, Any]:
    """Plan one globally aligned fresh AV window without assigning workflow intent."""

    _, _, previous_frames = validate_av_latent(previous_av_latent, name="previous_av_latent")
    overlap = int(overlap_frames)
    extension = int(extension_frames)
    if not is_h3_frame_count(overlap):
        raise ValueError("overlap_frames must satisfy the H3 17k+5 grid")
    if overlap > previous_frames:
        raise ValueError("overlap_frames cannot exceed the previous latent")
    if extension < 17 or extension % 17:
        raise ValueError("extension_frames must be a positive multiple of 17")

    window_frames = overlap + extension
    if not is_h3_frame_count(window_frames):
        raise ValueError("the planned window must satisfy the H3 17k+5 grid")
    window_start = previous_frames - overlap
    window_end = previous_frames + extension
    overlap_video_tokens = h3_visual_latent_frames(overlap)
    target_video_tokens = h3_visual_latent_frames(window_frames)
    extension_video_tokens = target_video_tokens - overlap_video_tokens
    overlap_audio_tokens = h3_audio_token_boundary(previous_frames) - h3_audio_token_boundary(
        window_start
    )
    extension_audio_tokens = h3_audio_token_boundary(window_end) - h3_audio_token_boundary(
        previous_frames
    )
    target_audio_tokens = overlap_audio_tokens + extension_audio_tokens
    payload: dict[str, Any] = {
        "schema": AV_WINDOW_LAYOUT_SCHEMA,
        "previous_frame_count": previous_frames,
        "window_start_frame": window_start,
        "window_end_frame": window_end,
        "window_frame_count": window_frames,
        "overlap_frames": overlap,
        "extension_frames": extension,
        "target_video_tokens": target_video_tokens,
        "target_audio_tokens": target_audio_tokens,
        "overlap_video_tokens": overlap_video_tokens,
        "overlap_audio_tokens": overlap_audio_tokens,
        "extension_video_tokens": extension_video_tokens,
        "extension_audio_tokens": extension_audio_tokens,
    }
    payload["layout_hash"] = content_hash(payload)
    return payload


def extract_h3_visual_stream(
    latent: Mapping[str, Any],
    *,
    timeline_origin_frame: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Expose the visual tensor as a standard LATENT while preserving the AV carrier.

    This is a deterministic adapter for visual-only latent tools. The caller must
    graft the result back onto the original AV carrier with
    ``replace_h3_video_stream`` so structural audio and duration remain explicit.
    """

    video, _, frame_count = validate_av_latent(
        latent,
        timeline_origin_frame=int(timeline_origin_frame),
        name="latent",
    )
    visual = {"samples": _clone(video)}
    payload: dict[str, Any] = {
        "schema": "cauce.h3-visual-stream-report/1",
        "timeline_origin_frame": int(timeline_origin_frame),
        "frame_count": frame_count,
        "video_shape": [int(item) for item in video.shape],
        "audio_preserved_in_source_carrier": True,
        "requires_explicit_graft": True,
    }
    payload["report_hash"] = content_hash(payload)
    return visual, payload


def validate_av_window_layout(layout: Mapping[str, Any]) -> None:
    if not isinstance(layout, Mapping) or layout.get("schema") != AV_WINDOW_LAYOUT_SCHEMA:
        raise ValueError(f"AV window layout must use schema {AV_WINDOW_LAYOUT_SCHEMA}")
    supplied_hash = layout.get("layout_hash")
    unhashed = {key: value for key, value in layout.items() if key != "layout_hash"}
    if supplied_hash != content_hash(unhashed):
        raise ValueError("AV window layout hash does not match its contents")

    fields = (
        "previous_frame_count",
        "window_start_frame",
        "window_end_frame",
        "window_frame_count",
        "overlap_frames",
        "extension_frames",
        "target_video_tokens",
        "target_audio_tokens",
        "overlap_video_tokens",
        "overlap_audio_tokens",
        "extension_video_tokens",
        "extension_audio_tokens",
    )
    try:
        values = {field: int(layout[field]) for field in fields}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("AV window layout is missing an integer field") from exc
    previous = values["previous_frame_count"]
    overlap = values["overlap_frames"]
    extension = values["extension_frames"]
    if not is_h3_frame_count(previous):
        raise ValueError("AV window previous_frame_count must satisfy the H3 17k+5 grid")
    if not is_h3_frame_count(overlap) or overlap > previous:
        raise ValueError("AV window overlap must be a valid H3 run within the previous latent")
    if extension < 17 or extension % 17:
        raise ValueError("AV window extension must be a positive multiple of 17")
    window = overlap + extension
    start = previous - overlap
    end = previous + extension
    expected = {
        "window_start_frame": start,
        "window_end_frame": end,
        "window_frame_count": window,
        "target_video_tokens": h3_visual_latent_frames(window),
        "target_audio_tokens": h3_audio_token_boundary(end) - h3_audio_token_boundary(start),
        "overlap_video_tokens": h3_visual_latent_frames(overlap),
        "overlap_audio_tokens": h3_audio_token_boundary(previous) - h3_audio_token_boundary(start),
    }
    expected["extension_video_tokens"] = (
        expected["target_video_tokens"] - expected["overlap_video_tokens"]
    )
    expected["extension_audio_tokens"] = h3_audio_token_boundary(end) - h3_audio_token_boundary(
        previous
    )
    for field, expected_value in expected.items():
        if values[field] != expected_value:
            raise ValueError(
                f"AV window layout {field} must be {expected_value}, got {values[field]}"
            )


def allocate_av_window_like(
    previous_av_latent: Mapping[str, Any],
    layout: Mapping[str, Any],
    *,
    nested_factory: NestedFactory | None = None,
) -> dict[str, Any]:
    """Allocate the zero AV target described by a validated absolute layout."""

    validate_av_window_layout(layout)
    video, audio, previous_frames = validate_av_latent(
        previous_av_latent,
        name="previous_av_latent",
    )
    if previous_frames != int(layout["previous_frame_count"]):
        raise ValueError("previous latent length differs from the AV window layout")
    target_video = _new_zeros(
        video,
        (
            int(video.shape[0]),
            int(video.shape[1]),
            int(layout["target_video_tokens"]),
            int(video.shape[3]),
            int(video.shape[4]),
        ),
    )
    target_audio = _new_zeros(
        audio,
        (
            int(audio.shape[0]),
            int(audio.shape[1]),
            int(audio.shape[2]),
            int(layout["target_audio_tokens"]),
        ),
    )
    target = _make_latent(target_video, target_audio, nested_factory)
    validate_av_latent(
        target,
        timeline_origin_frame=int(layout["window_start_frame"]),
        name="allocated_window",
    )
    return target
