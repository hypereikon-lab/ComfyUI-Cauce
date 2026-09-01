"""Continuous temporal and spatial denoise-mask algebra for H3 AV state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts import content_hash
from ..h3 import validate_av_latent
from ..timebase import (
    h3_audio_token_boundary,
    visual_token_boundary,
    visual_token_spans,
)
from .backend import (
    broadcast_audio_mask as _broadcast_audio_mask,
)
from .backend import (
    broadcast_video_mask as _broadcast_video_mask,
)
from .backend import (
    combine_mask as _combine_mask,
)
from .backend import (
    interval_weight as _interval_weight,
)
from .backend import (
    mask_all_finite as _mask_all_finite,
)
from .backend import (
    mask_digest as _mask_digest,
)
from .backend import (
    mask_min_max as _mask_min_max,
)
from .backend import (
    mask_streams as _mask_streams,
)
from .backend import (
    mask_tensor as _mask_tensor,
)
from .backend import (
    new_full as _new_full,
)
from .backend import (
    profile_tensor as _profile_tensor,
)
from .backend import (
    resize_mask_frames as _resize_mask_frames,
)
from .backend import (
    temporal_amax as _temporal_amax,
)
from .types import NestedFactory


def apply_av_denoise_interval(
    latent: Mapping[str, Any],
    *,
    start_frame: int,
    frame_count: int,
    timeline_origin_frame: int = 0,
    inside_strength_video: float = 1.0,
    outside_strength_video: float = 0.0,
    inside_strength_audio: float = 1.0,
    outside_strength_audio: float = 0.0,
    fade_in_frames: int = 0,
    fade_out_frames: int = 0,
    curve: str = "smoothstep",
    combine: str = "replace",
    nested_factory: NestedFactory | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Attach synchronized continuous H3 video/audio denoise masks.

    Strength 1 means generate and strength 0 means preserve. Temporal ramps are
    evaluated on each stream's own token centers, so the 24 fps visual lattice
    and absolute 40 Hz structural-audio clock stay synchronized without being
    forced onto a shared tensor axis.
    """

    origin = int(timeline_origin_frame)
    video, audio, total_frames = validate_av_latent(
        latent,
        timeline_origin_frame=origin,
    )
    start = int(start_frame)
    count = int(frame_count)
    end = start + count
    fade_in = int(fade_in_frames)
    fade_out = int(fade_out_frames)
    if start < 0 or count < 1 or end > total_frames:
        raise ValueError("denoise interval must be a non-empty range inside the AV latent")
    if fade_in < 0 or fade_out < 0:
        raise ValueError("fade frame counts cannot be negative")
    # Interval replacement/extraction must remain possible after sampling.
    visual_token_boundary(start)
    visual_token_boundary(end)
    strengths = (
        float(inside_strength_video),
        float(outside_strength_video),
        float(inside_strength_audio),
        float(outside_strength_audio),
    )
    if any(value < 0.0 or value > 1.0 for value in strengths):
        raise ValueError("AV denoise strengths must lie in [0, 1]")
    if combine not in {"replace", "maximum", "minimum", "multiply"}:
        raise ValueError("mask combine mode must be replace, maximum, minimum, or multiply")

    video_profile: list[float] = []
    for token_start, token_end in visual_token_spans(int(video.shape[2])):
        center = (token_start + token_end) / 2.0
        weight = _interval_weight(center, start, end, fade_in, fade_out, curve)
        video_profile.append(strengths[1] + (strengths[0] - strengths[1]) * weight)

    audio_origin_token = h3_audio_token_boundary(origin)
    audio_profile: list[float] = []
    for local_token in range(int(audio.shape[-1])):
        global_token_center = audio_origin_token + local_token + 0.5
        local_frame_center = global_token_center * (24.0 / 40.0) - origin
        weight = _interval_weight(
            local_frame_center,
            start,
            end,
            fade_in,
            fade_out,
            curve,
        )
        audio_profile.append(strengths[3] + (strengths[2] - strengths[3]) * weight)

    proposed_video = _broadcast_video_mask(
        _profile_tensor(video, video_profile),
        video,
    )
    proposed_audio = _broadcast_audio_mask(
        _profile_tensor(audio, audio_profile),
        audio,
    )
    existing_video, existing_audio = _mask_streams(latent)
    mask_video = _combine_mask(existing_video, proposed_video, combine)
    mask_audio = _combine_mask(existing_audio, proposed_audio, combine)
    out = dict(latent)
    out["noise_mask"] = (
        nested_factory((mask_video, mask_audio))
        if nested_factory is not None
        else (mask_video, mask_audio)
    )

    rounded_video = [round(value, 8) for value in video_profile]
    rounded_audio = [round(value, 8) for value in audio_profile]
    report: dict[str, Any] = {
        "schema": "cauce.h3-av-denoise-interval-report/1",
        "timeline_origin_frame": origin,
        "target_frame_count": total_frames,
        "denoise_range": [start, end],
        "fade_in_frames": fade_in,
        "fade_out_frames": fade_out,
        "curve": curve,
        "combine": combine,
        "video_strength": {"inside": strengths[0], "outside": strengths[1]},
        "audio_strength": {"inside": strengths[2], "outside": strengths[3]},
        "video_profile": {
            "tokens": len(video_profile),
            "minimum": min(video_profile),
            "maximum": max(video_profile),
            "hash": content_hash(rounded_video),
        },
        "audio_profile": {
            "tokens": len(audio_profile),
            "minimum": min(audio_profile),
            "maximum": max(audio_profile),
            "hash": content_hash(rounded_audio),
        },
        "requires_comfyui_core": "ff6c8a8af144fc9e9e7bc436b1b202f9316848d8-or-newer",
    }
    report["mask_hash"] = content_hash(report)
    return out, report


def apply_video_denoise_mask(
    latent: Mapping[str, Any],
    mask: Any,
    *,
    start_frame: int,
    frame_count: int,
    timeline_origin_frame: int = 0,
    inside_strength_video: float = 1.0,
    outside_strength_video: float = 0.0,
    audio_strength: float = 0.0,
    combine: str = "replace",
    nested_factory: NestedFactory | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project a decoded spatial/video mask onto H3's visual-token lattice.

    ``mask`` is either one static ``[H,W]``/``[1,H,W]`` mask or one mask per
    decoded frame in the requested interval. Spatial resizing remains
    continuous; multiple decoded masks covered by one H3 visual token are
    reduced with ``amax``. The interval boundaries must be representable on the
    native visual-token clock. Structural audio receives one explicit constant
    strength because a spatial mask has no audio geometry.
    """

    origin = int(timeline_origin_frame)
    video, audio, total_frames = validate_av_latent(
        latent,
        timeline_origin_frame=origin,
    )
    start = int(start_frame)
    count = int(frame_count)
    end = start + count
    if start < 0 or count < 1 or end > total_frames:
        raise ValueError("video denoise mask interval must lie inside the AV latent")
    video_start_token = visual_token_boundary(start)
    video_end_token = visual_token_boundary(end)
    strengths = (
        float(inside_strength_video),
        float(outside_strength_video),
        float(audio_strength),
    )
    if any(value < 0.0 or value > 1.0 for value in strengths):
        raise ValueError("video-mask denoise strengths must lie in [0, 1]")
    if combine not in {"replace", "maximum", "minimum", "multiply"}:
        raise ValueError("mask combine mode must be replace, maximum, minimum, or multiply")

    source = _mask_tensor(video, mask)
    if getattr(source, "ndim", 0) == 2:
        source = source.reshape((1,) + tuple(source.shape))
    if getattr(source, "ndim", 0) != 3:
        raise ValueError("video denoise mask must have shape [H,W] or [frames,H,W]")
    source_frames = int(source.shape[0])
    if source_frames not in {1, count}:
        raise ValueError(
            "video denoise mask must contain one static mask or exactly frame_count masks"
        )
    if int(source.shape[1]) < 1 or int(source.shape[2]) < 1:
        raise ValueError("video denoise mask spatial dimensions must be positive")
    if not _mask_all_finite(source):
        raise ValueError("video denoise mask values must be finite")
    source_min, source_max = _mask_min_max(source)
    if source_min < 0.0 or source_max > 1.0:
        raise ValueError("video denoise mask values must lie in [0, 1]")

    resized = _resize_mask_frames(source, int(video.shape[3]), int(video.shape[4]))
    proposed_video = _new_full(
        video,
        (
            int(video.shape[0]),
            1,
            int(video.shape[2]),
            int(video.shape[3]),
            int(video.shape[4]),
        ),
        strengths[1],
    )
    token_spans = visual_token_spans(int(video.shape[2]))
    for token_index in range(video_start_token, video_end_token):
        token_start, token_end = token_spans[token_index]
        if source_frames == 1:
            spatial = resized[0]
        else:
            spatial = _temporal_amax(
                resized,
                token_start - start,
                token_end - start,
            )
        proposed_video[:, :, token_index] = strengths[1] + (strengths[0] - strengths[1]) * spatial

    proposed_audio = _new_full(
        audio,
        (int(audio.shape[0]), 1, int(audio.shape[2]), int(audio.shape[3])),
        strengths[2],
    )
    existing_video, existing_audio = _mask_streams(latent)
    mask_video = _combine_mask(existing_video, proposed_video, combine)
    mask_audio = _combine_mask(existing_audio, proposed_audio, combine)
    out = dict(latent)
    out["noise_mask"] = (
        nested_factory((mask_video, mask_audio))
        if nested_factory is not None
        else (mask_video, mask_audio)
    )

    result_min, result_max = _mask_min_max(mask_video)
    report: dict[str, Any] = {
        "schema": "cauce.h3-video-denoise-mask-report/1",
        "timeline_origin_frame": origin,
        "target_frame_count": total_frames,
        "mask_frame_range": [start, end],
        "video_token_range": [video_start_token, video_end_token],
        "source_mask_shape": [int(item) for item in source.shape],
        "latent_mask_shape": [int(item) for item in proposed_video.shape],
        "temporal_projection": ("static" if source_frames == 1 else "amax-per-h3-visual-token"),
        "spatial_projection": "continuous-bilinear-to-video-latent-grid",
        "combine": combine,
        "video_strength": {"inside": strengths[0], "outside": strengths[1]},
        "audio_strength": strengths[2],
        "source_minimum": source_min,
        "source_maximum": source_max,
        "result_minimum": result_min,
        "result_maximum": result_max,
        "result_digest": _mask_digest(mask_video),
        "requires_comfyui_core": "ff6c8a8af144fc9e9e7bc436b1b202f9316848d8-or-newer",
    }
    report["mask_hash"] = content_hash(report)
    return out, report


def clear_av_denoise_mask(
    latent: Mapping[str, Any],
    *,
    timeline_origin_frame: int = 0,
) -> tuple[dict[str, Any], bool]:
    """Remove a spent sampler noise mask without changing either AV stream."""

    validate_av_latent(latent, timeline_origin_frame=int(timeline_origin_frame))
    out = dict(latent)
    removed = out.pop("noise_mask", None) is not None
    return out, removed
