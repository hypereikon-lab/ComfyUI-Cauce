"""Low-level, timeline-aware operations for packed MiniMax H3 AV latents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

from .contracts import AV_SPAN_SCHEMA, AV_WINDOW_LAYOUT_SCHEMA, content_hash
from .h3 import get_av_streams, validate_av_latent
from .timebase import (
    h3_audio_token_boundary,
    h3_visual_latent_frames,
    is_h3_frame_count,
    visual_token_boundary,
)


NestedFactory = Callable[[tuple[Any, Any]], Any]


def _clone(value: Any):
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    copy = getattr(value, "copy", None)
    if callable(copy):
        return copy()
    raise TypeError("AV tensors must provide clone() or copy()")


def _new_zeros(reference: Any, shape: tuple[int, ...]):
    new_zeros = getattr(reference, "new_zeros", None)
    if callable(new_zeros):
        return new_zeros(shape)
    try:
        import numpy as np

        return np.zeros(shape, dtype=reference.dtype)
    except (ImportError, AttributeError) as exc:  # pragma: no cover - NumPy ships with ComfyUI
        raise TypeError("AV tensors must support new_zeros() or be NumPy arrays") from exc


def _concatenate(values: tuple[Any, ...], axis: int):
    first = values[0]
    try:
        import torch

        if isinstance(first, torch.Tensor):
            return torch.cat(values, dim=axis)
    except ImportError:  # pragma: no cover - PyTorch ships with ComfyUI
        pass
    try:
        import numpy as np

        if isinstance(first, np.ndarray):
            return np.concatenate(values, axis=axis)
    except ImportError:  # pragma: no cover - NumPy ships with ComfyUI
        pass
    raise TypeError("AV tensors must be PyTorch tensors or NumPy arrays")


def _make_latent(video: Any, audio: Any, nested_factory: NestedFactory | None):
    samples = nested_factory((video, audio)) if nested_factory is not None else (video, audio)
    return {"samples": samples}


def _dtype(value: Any) -> str:
    return str(getattr(value, "dtype", "unknown"))


def _device(value: Any) -> str:
    return str(getattr(value, "device", "cpu"))


def _validate_tensor_compatibility(
    left_video: Any,
    left_audio: Any,
    right_video: Any,
    right_audio: Any,
) -> None:
    if tuple(left_video.shape[:2]) != tuple(right_video.shape[:2]):
        raise ValueError("H3 video batch/channel dimensions must match")
    if tuple(left_video.shape[3:]) != tuple(right_video.shape[3:]):
        raise ValueError("H3 video spatial dimensions must match")
    if tuple(left_audio.shape[:3]) != tuple(right_audio.shape[:3]):
        raise ValueError("H3 structural-audio batch/channel dimensions must match")
    if getattr(left_video, "dtype", None) != getattr(right_video, "dtype", None):
        raise TypeError("H3 video dtypes must match")
    if getattr(left_audio, "dtype", None) != getattr(right_audio, "dtype", None):
        raise TypeError("H3 structural-audio dtypes must match")
    if _device(left_video) != _device(right_video):
        raise ValueError("H3 video devices must match")
    if _device(left_audio) != _device(right_audio):
        raise ValueError("H3 structural-audio devices must match")


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
        "target_audio_tokens": h3_audio_token_boundary(end)
        - h3_audio_token_boundary(start),
        "overlap_video_tokens": h3_visual_latent_frames(overlap),
        "overlap_audio_tokens": h3_audio_token_boundary(previous)
        - h3_audio_token_boundary(start),
    }
    expected["extension_video_tokens"] = (
        expected["target_video_tokens"] - expected["overlap_video_tokens"]
    )
    expected["extension_audio_tokens"] = h3_audio_token_boundary(
        end
    ) - h3_audio_token_boundary(previous)
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
        "audio_end_token": h3_audio_token_boundary(origin + end)
        - h3_audio_token_boundary(origin),
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
