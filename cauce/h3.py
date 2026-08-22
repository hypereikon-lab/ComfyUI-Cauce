"""H3-specific validation and compatibility helpers.

No ComfyUI symbols are imported at module import time.  The adapter functions
load the current official nodes only when a graph executes inside ComfyUI.
"""

from __future__ import annotations

import copy
import importlib
from typing import Any

from .contracts import WINDOW_SCHEMA, range_fraction
from .timebase import (
    H3_FPS,
    as_fraction,
    fraction_from_payload,
    frames_to_seconds,
    round_fraction,
    visual_span_for_tokens,
)


REF_LIMITS = {"images": 9, "videos": 3, "audios": 3, "total": 12}
REF_MEDIA_MIN_SECONDS = 2
REF_MEDIA_MAX_SECONDS = 15
REF_MEDIA_TOTAL_SECONDS = 15


def empty_reference_set() -> dict[str, Any]:
    return {
        "schema": "cauce.h3-reference-set/1",
        "images": [],
        "videos": [],
        "video_audios": [],
        "audios": [],
    }


def append_reference(
    reference_set: dict[str, Any] | None,
    *,
    kind: str,
    media: Any,
    audio: Any = None,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    out = copy.copy(reference_set or empty_reference_set())
    if out.get("schema") != "cauce.h3-reference-set/1":
        raise ValueError("invalid CAUCE H3 reference set")
    for key in ("images", "videos", "video_audios", "audios"):
        out[key] = list(out.get(key, []))

    if kind == "image":
        if len(out["images"]) >= REF_LIMITS["images"]:
            raise ValueError("Ref2VA accepts at most 9 image references")
        out["images"].append(media)
    elif kind == "video":
        if len(out["videos"]) >= REF_LIMITS["videos"]:
            raise ValueError("Ref2VA accepts at most 3 video references")
        out["videos"].append(media)
        out["video_audios"].append(audio)
    elif kind == "audio":
        if len(out["audios"]) >= REF_LIMITS["audios"]:
            raise ValueError("Ref2VA accepts at most 3 standalone audio references")
        out["audios"].append(media)
    else:
        raise ValueError("reference kind must be image, video, or audio")

    metadata = list(out.get("metadata", []))
    metadata.append({"kind": kind, "duration_seconds": duration_seconds})
    out["metadata"] = metadata
    validate_reference_set(out)
    return out


def validate_reference_set(reference_set: dict[str, Any]) -> None:
    images = list(reference_set.get("images", []))
    videos = list(reference_set.get("videos", []))
    audios = list(reference_set.get("audios", []))
    if len(images) > REF_LIMITS["images"]:
        raise ValueError("Ref2VA accepts at most 9 image references")
    if len(videos) > REF_LIMITS["videos"]:
        raise ValueError("Ref2VA accepts at most 3 video references")
    if len(audios) > REF_LIMITS["audios"]:
        raise ValueError("Ref2VA accepts at most 3 standalone audio references")
    if len(images) + len(videos) + len(audios) > REF_LIMITS["total"]:
        raise ValueError("Ref2VA accepts at most 12 total references")

    durations = [
        float(item["duration_seconds"])
        for item in reference_set.get("metadata", [])
        if item.get("kind") in {"video", "audio"}
        and item.get("duration_seconds") is not None
    ]
    for duration in durations:
        if not REF_MEDIA_MIN_SECONDS <= duration <= REF_MEDIA_MAX_SECONDS:
            raise ValueError("Ref2VA video/audio references must be 2-15 seconds")
    if sum(durations) > REF_MEDIA_TOTAL_SECONDS:
        raise ValueError("Ref2VA video/audio reference duration exceeds 15 seconds total")


def reference_tags(reference_set: dict[str, Any]) -> str:
    tags: list[str] = []
    tags.extend(f"<Picture {index}>" for index in range(1, len(reference_set.get("images", [])) + 1))
    tags.extend(f"<Video {index}>" for index in range(1, len(reference_set.get("videos", [])) + 1))
    tags.extend(f"<Audio {index}>" for index in range(1, len(reference_set.get("audios", [])) + 1))
    return ", ".join(tags)


def frame_index_in_window(window: dict[str, Any], master_time: object) -> int:
    if window.get("schema") != WINDOW_SCHEMA:
        raise ValueError(f"window schema must be {WINDOW_SCHEMA}")
    render_start, render_end = range_fraction(window["render_range"])
    time_f = fraction_from_payload(master_time)
    if time_f < render_start or time_f >= render_end:
        raise ValueError("guide time lies outside the render window")
    frame = round_fraction((time_f - render_start) * H3_FPS)
    frame_count = int(window["shape"]["pixel_frames"])
    if frame < 0 or frame >= frame_count:
        raise ValueError("guide resolves outside the H3 frame range")
    return frame


def window_local_seconds(window: dict[str, Any], master_time: object):
    render_start, _ = range_fraction(window["render_range"])
    return fraction_from_payload(master_time) - render_start


def get_av_streams(latent: dict[str, Any]) -> tuple[Any, Any]:
    samples = latent.get("samples")
    if samples is None:
        raise ValueError("latent has no samples")
    if getattr(samples, "is_nested", False):
        streams = list(samples.unbind())
    elif isinstance(samples, (list, tuple)):
        streams = list(samples)
    else:
        raise ValueError("expected a nested MiniMax H3 audiovisual latent")
    if len(streams) < 2:
        raise ValueError("H3 latent must contain video and audio streams")
    video, audio = streams[0], streams[1]
    if getattr(video, "ndim", 0) != 5 or getattr(audio, "ndim", 0) != 4:
        raise ValueError("unexpected H3 AV latent shapes")
    return video, audio


def pixel_frames_from_video_latent(video: Any) -> int:
    return visual_span_for_tokens(int(video.shape[2]))


def official_h3_nodes():
    try:
        module = importlib.import_module("comfy_extras.nodes_minimax_h3")
        image_to_video = getattr(module, "MiniMaxH3ImageToVideo")
        reference_to_video = getattr(module, "MiniMaxH3ReferenceToVideo")
        add_guide = getattr(module, "MiniMaxH3AddGuide", None)
    except (ImportError, AttributeError) as exc:  # pragma: no cover - inside ComfyUI
        raise RuntimeError(
            "CAUCE requires native MiniMaxH3ImageToVideo and "
            f"MiniMaxH3ReferenceToVideo nodes ({type(exc).__name__}: {exc})"
        ) from exc
    return image_to_video, reference_to_video, add_guide


def unwrap_node_output(value: Any) -> tuple[Any, ...]:
    result = getattr(value, "result", value)
    if isinstance(result, tuple):
        return result
    if isinstance(result, list):
        return tuple(result)
    return (result,)


def execute_fl2va(**kwargs: Any) -> tuple[Any, Any]:
    image_to_video, _, _ = official_h3_nodes()
    result = unwrap_node_output(image_to_video.execute(**kwargs))
    if len(result) != 2:
        raise RuntimeError("unexpected output from the official H3 FL2VA node")
    return result[0], result[1]


def execute_ref2va(reference_set: dict[str, Any] | None = None, **kwargs: Any) -> tuple[Any, Any]:
    _, reference_to_video, _ = official_h3_nodes()
    refs = reference_set or empty_reference_set()
    validate_reference_set(refs)
    images = {f"ref_image_{index}": value for index, value in enumerate(refs["images"], 1)}
    videos = {f"ref_video_{index}": value for index, value in enumerate(refs["videos"], 1)}
    video_audios = {
        f"ref_video_audio_{index}": value
        for index, value in enumerate(refs["video_audios"], 1)
        if value is not None
    }
    audios = {f"ref_audio_{index}": value for index, value in enumerate(refs["audios"], 1)}
    result = unwrap_node_output(
        reference_to_video.execute(
            ref_images=images,
            ref_videos=videos,
            ref_video_audios=video_audios,
            ref_audios=audios,
            **kwargs,
        )
    )
    if len(result) != 2:
        raise RuntimeError("unexpected output from the official H3 Ref2VA node")
    return result[0], result[1]


def execute_add_guide(**kwargs: Any) -> Any:
    _, _, add_guide = official_h3_nodes()
    if add_guide is None:
        raise RuntimeError(
            "this ComfyUI build does not provide MiniMaxH3AddGuide; "
            "FL2VA and Ref2VA remain available, but timed guides require a "
            "newer official H3 runtime"
        )
    result = unwrap_node_output(add_guide.execute(**kwargs))
    if len(result) != 1:
        raise RuntimeError("unexpected output from the official H3 AddGuide node")
    return result[0]


def duration_from_frames(frames: int) -> float:
    return float(frames_to_seconds(frames))
