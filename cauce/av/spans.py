"""Extraction, placement, branching, and composition of native H3 AV spans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..contracts import AV_SPAN_SCHEMA, content_hash
from ..h3 import validate_av_latent
from ..timebase import (
    h3_audio_token_boundary,
    is_h3_frame_count,
    visual_token_boundary,
)
from .backend import (
    clone as _clone,
)
from .backend import (
    concatenate as _concatenate,
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
    validate_tensor_compatibility as _validate_tensor_compatibility,
)
from .backend import (
    with_streams as _with_streams,
)
from .layout import validate_av_window_layout
from .types import NestedFactory


def extract_av_span(
    latent: Mapping[str, Any],
    *,
    start_frame: int,
    frame_count: int,
    timeline_origin_frame: int = 0,
) -> dict[str, Any]:
    """Extract one synchronized video/audio span at exact token boundaries."""

    origin = int(timeline_origin_frame)
    start = int(start_frame)
    count = int(frame_count)
    video, audio, total_frames = validate_av_latent(
        latent,
        timeline_origin_frame=origin,
    )
    if start < 0 or count < 1:
        raise ValueError("AV span requires start_frame >= 0 and frame_count >= 1")
    end = start + count
    if end > total_frames:
        raise ValueError(f"AV latent has {total_frames} frames but the span ends at {end}")
    video_start = visual_token_boundary(start)
    video_end = visual_token_boundary(end)
    global_start = origin + start
    global_end = origin + end
    audio_start = h3_audio_token_boundary(global_start) - h3_audio_token_boundary(origin)
    audio_end = h3_audio_token_boundary(global_end) - h3_audio_token_boundary(origin)
    descriptor: dict[str, Any] = {
        "timeline_origin_frame": origin,
        "local_start_frame": start,
        "local_end_frame": end,
        "global_start_frame": global_start,
        "global_end_frame": global_end,
        "frame_count": count,
        "video_start_token": video_start,
        "video_end_token": video_end,
        "video_tokens": video_end - video_start,
        "audio_start_token": audio_start,
        "audio_end_token": audio_end,
        "audio_tokens": audio_end - audio_start,
        "video_spatial_shape": [int(video.shape[3]), int(video.shape[4])],
        "video_dtype": _dtype(video),
        "audio_dtype": _dtype(audio),
        "video_device": _device(video),
        "audio_device": _device(audio),
    }
    descriptor_hash = content_hash(descriptor)
    return {
        "schema": AV_SPAN_SCHEMA,
        "descriptor": descriptor,
        "descriptor_hash": descriptor_hash,
        "video": _clone(video[:, :, video_start:video_end]),
        "audio": _clone(audio[..., audio_start:audio_end]),
    }


def validate_av_span(span: Mapping[str, Any]) -> tuple[Any, Any, Mapping[str, Any]]:
    if not isinstance(span, Mapping) or span.get("schema") != AV_SPAN_SCHEMA:
        raise ValueError(f"AV span must use schema {AV_SPAN_SCHEMA}")
    descriptor = span.get("descriptor")
    if not isinstance(descriptor, Mapping):
        raise ValueError("AV span descriptor is missing")
    if span.get("descriptor_hash") != content_hash(descriptor):
        raise ValueError("AV span descriptor hash does not match its contents")
    video = span.get("video")
    audio = span.get("audio")
    if getattr(video, "ndim", 0) != 5 or getattr(audio, "ndim", 0) != 4:
        raise ValueError("AV span tensors have unexpected shapes")
    if int(video.shape[0]) != 1 or int(audio.shape[0]) != 1:
        raise ValueError("AV span video and audio batch sizes must both be 1")
    if int(video.shape[1]) != 24 or int(audio.shape[1]) != 32 or int(audio.shape[2]) != 2:
        raise ValueError("AV span tensors do not have MiniMax H3 channel geometry")
    fields = (
        "timeline_origin_frame",
        "local_start_frame",
        "local_end_frame",
        "global_start_frame",
        "global_end_frame",
        "frame_count",
        "video_start_token",
        "video_end_token",
        "video_tokens",
        "audio_start_token",
        "audio_end_token",
        "audio_tokens",
    )
    try:
        values = {field: int(descriptor[field]) for field in fields}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("AV span descriptor is missing an integer field") from exc
    origin = values["timeline_origin_frame"]
    start = values["local_start_frame"]
    end = values["local_end_frame"]
    count = values["frame_count"]
    if origin < 0 or start < 0 or count < 1 or end != start + count:
        raise ValueError("AV span descriptor has an invalid local frame range")
    expected = {
        "global_start_frame": origin + start,
        "global_end_frame": origin + end,
        "video_start_token": visual_token_boundary(start),
        "video_end_token": visual_token_boundary(end),
        "audio_start_token": h3_audio_token_boundary(origin + start)
        - h3_audio_token_boundary(origin),
        "audio_end_token": h3_audio_token_boundary(origin + end) - h3_audio_token_boundary(origin),
    }
    expected["video_tokens"] = expected["video_end_token"] - expected["video_start_token"]
    expected["audio_tokens"] = expected["audio_end_token"] - expected["audio_start_token"]
    for field, expected_value in expected.items():
        if values[field] != expected_value:
            raise ValueError(
                f"AV span descriptor {field} must be {expected_value}, got {values[field]}"
            )
    if int(video.shape[2]) != values["video_tokens"]:
        raise ValueError("AV span video tensor differs from its descriptor")
    if int(audio.shape[-1]) != values["audio_tokens"]:
        raise ValueError("AV span audio tensor differs from its descriptor")
    if list(descriptor.get("video_spatial_shape", ())) != [
        int(video.shape[3]),
        int(video.shape[4]),
    ]:
        raise ValueError("AV span video spatial shape differs from its descriptor")
    if descriptor.get("video_dtype") != _dtype(video) or descriptor.get("audio_dtype") != _dtype(
        audio
    ):
        raise ValueError("AV span tensor dtype differs from its descriptor")
    if descriptor.get("video_device") != _device(video) or descriptor.get(
        "audio_device"
    ) != _device(audio):
        raise ValueError("AV span tensor device differs from its descriptor")
    return video, audio, descriptor


def build_av_span_keyframes(
    existing_keyframes: Sequence[Mapping[str, Any]],
    span: Mapping[str, Any],
    target_av_latent: Mapping[str, Any],
    target_layout: Mapping[str, Any],
    *,
    target_frame_idx: int,
) -> list[dict[str, Any]]:
    """Return keyframe metadata with one compatible latent AV span inserted."""

    validate_av_window_layout(target_layout)
    target_origin = int(target_layout["window_start_frame"])
    target_video, target_audio, target_frames = validate_av_latent(
        target_av_latent,
        timeline_origin_frame=target_origin,
        name="target_av_latent",
    )
    span_video, span_audio, descriptor = validate_av_span(span)
    _validate_tensor_compatibility(
        target_video,
        target_audio,
        span_video,
        span_audio,
    )
    index = int(target_frame_idx)
    span_frames = int(descriptor["frame_count"])
    if index < 0 or index + span_frames > target_frames:
        raise ValueError("latent AV guide must fit completely inside the target window")
    target_global_start = target_origin + index
    expected_audio_tokens = h3_audio_token_boundary(
        target_global_start + span_frames
    ) - h3_audio_token_boundary(target_global_start)
    if int(span_audio.shape[-1]) != expected_audio_tokens:
        raise ValueError(
            "latent AV guide audio length is not aligned at the requested target frame"
        )

    keyframes = [dict(item) for item in existing_keyframes]
    for keyframe in keyframes:
        position = keyframe.get("resolved_frame_index")
        if not isinstance(position, (int, float)):
            raise ValueError("existing H3 guides must expose resolved_frame_index")
        if int(position) < 0 or int(position) >= target_frames:
            raise ValueError("existing H3 guide starts outside the target window")
    keyframes.append(
        {
            "resolved_frame_index": index,
            "latent": _clone(span_video),
            "audio_latent": _clone(span_audio),
        }
    )
    keyframes.sort(key=lambda item: int(item["resolved_frame_index"]))
    return keyframes


def place_av_span(
    target_av_latent: Mapping[str, Any],
    span: Mapping[str, Any],
    *,
    target_frame_idx: int,
    timeline_origin_frame: int = 0,
    nested_factory: NestedFactory | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy one exact native AV span into a target without assigning denoise policy.

    Placement may deliberately rebase a span onto another global frame. The two
    streams must still occupy the same number of visual and audio tokens at the
    requested target position; incompatible 24->40 Hz phases fail closed.
    """

    origin = int(timeline_origin_frame)
    target_video, target_audio, target_frames = validate_av_latent(
        target_av_latent,
        timeline_origin_frame=origin,
        name="target_av_latent",
    )
    span_video, span_audio, descriptor = validate_av_span(span)
    _validate_tensor_compatibility(
        target_video,
        target_audio,
        span_video,
        span_audio,
    )
    index = int(target_frame_idx)
    span_frames = int(descriptor["frame_count"])
    end = index + span_frames
    if index < 0 or end > target_frames:
        raise ValueError("native AV span must fit completely inside the target latent")

    video_start = visual_token_boundary(index)
    video_end = visual_token_boundary(end)
    audio_start = h3_audio_token_boundary(origin + index) - h3_audio_token_boundary(origin)
    audio_end = h3_audio_token_boundary(origin + end) - h3_audio_token_boundary(origin)
    if video_end - video_start != int(span_video.shape[2]):
        raise ValueError("native AV span video tokens do not align at the requested target frame")
    if audio_end - audio_start != int(span_audio.shape[-1]):
        raise ValueError("native AV span audio tokens do not align at the requested target frame")

    placed_video = _clone(target_video)
    placed_audio = _clone(target_audio)
    placed_video[:, :, video_start:video_end] = span_video
    placed_audio[..., audio_start:audio_end] = span_audio
    placed = _with_streams(
        target_av_latent,
        placed_video,
        placed_audio,
        nested_factory,
    )
    target_global_start = origin + index
    report: dict[str, Any] = {
        "schema": "cauce.h3-av-placement-report/1",
        "timeline_origin_frame": origin,
        "target_frame_range": [index, end],
        "target_global_range": [target_global_start, origin + end],
        "source_global_range": [
            int(descriptor["global_start_frame"]),
            int(descriptor["global_end_frame"]),
        ],
        "frame_count": span_frames,
        "video_token_range": [video_start, video_end],
        "audio_token_range": [audio_start, audio_end],
        "rebased": int(descriptor["global_start_frame"]) != target_global_start,
        "source_descriptor_hash": span["descriptor_hash"],
    }
    report["placement_hash"] = content_hash(report)
    return placed, report


def replace_av_span(
    base_av_latent: Mapping[str, Any],
    replacement_span: Mapping[str, Any],
    *,
    timeline_origin_frame: int = 0,
    nested_factory: NestedFactory | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace one globally aligned native AV interval without changing duration."""

    origin = int(timeline_origin_frame)
    _, _, total_frames = validate_av_latent(
        base_av_latent,
        timeline_origin_frame=origin,
        name="base_av_latent",
    )
    _, _, descriptor = validate_av_span(replacement_span)
    target_index = int(descriptor["global_start_frame"]) - origin
    if target_index < 0 or int(descriptor["global_end_frame"]) > origin + total_frames:
        raise ValueError("replacement AV span lies outside the base latent timeline")
    replaced, placement = place_av_span(
        base_av_latent,
        replacement_span,
        target_frame_idx=target_index,
        timeline_origin_frame=origin,
        nested_factory=nested_factory,
    )
    if placement["rebased"]:
        raise ValueError("replacement AV span must retain its original global frame range")
    replaced.pop("noise_mask", None)
    report: dict[str, Any] = {
        "schema": "cauce.h3-av-replacement-report/1",
        "timeline_origin_frame": origin,
        "target_frame_count": total_frames,
        "replaced_global_range": placement["target_global_range"],
        "source_descriptor_hash": replacement_span["descriptor_hash"],
        "placement_hash": placement["placement_hash"],
    }
    report["replacement_hash"] = content_hash(report)
    return replaced, report


def append_av_span(
    base_av_latent: Mapping[str, Any],
    span: Mapping[str, Any],
    *,
    nested_factory: NestedFactory | None = None,
) -> tuple[dict[str, Any], int]:
    """Append one globally contiguous AV span without resampling or overlap policy."""

    base_video, base_audio, base_frames = validate_av_latent(
        base_av_latent,
        name="base_av_latent",
    )
    span_video, span_audio, descriptor = validate_av_span(span)
    if int(descriptor["global_start_frame"]) != base_frames:
        raise ValueError(
            "AV span is not globally contiguous with the base latent: "
            f"expected {base_frames}, got {descriptor['global_start_frame']}"
        )
    _validate_tensor_compatibility(base_video, base_audio, span_video, span_audio)
    extended_video = _concatenate((base_video, span_video), axis=2)
    extended_audio = _concatenate((base_audio, span_audio), axis=-1)
    extended = _make_latent(extended_video, extended_audio, nested_factory)
    total_frames = int(descriptor["global_end_frame"])
    _, _, validated_frames = validate_av_latent(extended, name="extended_av_latent")
    if validated_frames != total_frames:
        raise ValueError("appended AV latent length differs from the span timeline")
    return extended, total_frames


def split_av_latent(
    latent: Mapping[str, Any],
    *,
    cut_frame: int,
    nested_factory: NestedFactory | None = None,
) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    """Split an origin-zero cumulative state into prefix latent and suffix span."""

    cut = int(cut_frame)
    video, audio, total_frames = validate_av_latent(latent)
    if not is_h3_frame_count(cut):
        raise ValueError("cut_frame must leave a complete 17k+5 H3 prefix")
    if cut >= total_frames:
        raise ValueError("cut_frame must leave a non-empty suffix")
    video_end = visual_token_boundary(cut)
    audio_end = h3_audio_token_boundary(cut)
    prefix = _make_latent(
        _clone(video[:, :, :video_end]),
        _clone(audio[..., :audio_end]),
        nested_factory,
    )
    _, _, prefix_frames = validate_av_latent(
        prefix,
        name="prefix_av_latent",
    )
    suffix_frames = total_frames - cut
    suffix = extract_av_span(
        latent,
        start_frame=cut,
        frame_count=suffix_frames,
    )
    return prefix, suffix, prefix_frames, suffix_frames
