"""Spatial canvas and native visual-token transformations for H3 AV state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts import content_hash
from ..h3 import validate_av_latent
from ..timebase import (
    ceil_h3_frame_count,
    h3_audio_token_boundary,
    h3_visual_latent_frames,
    is_h3_frame_count,
    visual_token_spans,
)
from .backend import (
    broadcast_video_mask as _broadcast_video_mask,
)
from .backend import (
    clone as _clone,
)
from .backend import (
    curve as _curve,
)
from .backend import (
    mask_digest as _mask_digest,
)
from .backend import (
    new_full as _new_full,
)
from .backend import (
    new_zeros as _new_zeros,
)
from .backend import (
    profile_tensor as _profile_tensor,
)
from .backend import (
    resize_spatial as _resize_spatial,
)
from .backend import (
    with_streams as _with_streams,
)
from .types import NestedFactory


def expand_av_canvas(
    latent: Mapping[str, Any],
    *,
    target_width: int,
    target_height: int,
    offset_x: int,
    offset_y: int,
    source_strength_video: float = 0.0,
    new_region_strength_video: float = 1.0,
    audio_strength: float = 0.0,
    timeline_origin_frame: int = 0,
    nested_factory: NestedFactory | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Place a packed H3 video latent on a larger 32-pixel-aligned canvas.

    The source visual state is copied exactly, new regions are zero allocated,
    and a nested denoise mask is attached. The synchronized structural-audio
    stream is cloned without resizing. Existing mask metadata is rejected so a
    caller must make mask lifecycle explicit before reframing.
    """

    origin = int(timeline_origin_frame)
    video, audio, total_frames = validate_av_latent(
        latent,
        timeline_origin_frame=origin,
    )
    if latent.get("noise_mask") is not None:
        raise ValueError("clear the existing AV denoise mask before expanding the canvas")
    width = int(target_width)
    height = int(target_height)
    x = int(offset_x)
    y = int(offset_y)
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise ValueError("target canvas dimensions must be positive multiples of 32 pixels")
    if x < 0 or y < 0 or x % 32 or y % 32:
        raise ValueError("canvas offsets must be non-negative multiples of 32 pixels")
    source_width = int(video.shape[4]) * 16
    source_height = int(video.shape[3]) * 16
    if source_width % 32 or source_height % 32:
        raise ValueError("source H3 latent must already align to the 32-pixel DiT patch grid")
    if width < source_width or height < source_height:
        raise ValueError("target canvas cannot be smaller than the source latent")
    if width == source_width and height == source_height:
        raise ValueError("target canvas must expand at least one source dimension")
    if x + source_width > width or y + source_height > height:
        raise ValueError("source latent placement does not fit inside the target canvas")
    strengths = (
        float(source_strength_video),
        float(new_region_strength_video),
        float(audio_strength),
    )
    if any(value < 0.0 or value > 1.0 for value in strengths):
        raise ValueError("canvas denoise strengths must lie in [0, 1]")

    target_h = height // 16
    target_w = width // 16
    offset_h = y // 16
    offset_w = x // 16
    expanded_video = _new_zeros(
        video,
        (
            int(video.shape[0]),
            int(video.shape[1]),
            int(video.shape[2]),
            target_h,
            target_w,
        ),
    )
    expanded_video[
        :,
        :,
        :,
        offset_h : offset_h + int(video.shape[3]),
        offset_w : offset_w + int(video.shape[4]),
    ] = video
    expanded_audio = _clone(audio)
    expanded = _with_streams(latent, expanded_video, expanded_audio, nested_factory)

    video_mask = _new_full(
        video,
        (int(video.shape[0]), 1, int(video.shape[2]), target_h, target_w),
        strengths[1],
    )
    video_mask[
        :,
        :,
        :,
        offset_h : offset_h + int(video.shape[3]),
        offset_w : offset_w + int(video.shape[4]),
    ] = strengths[0]
    audio_mask = _new_full(
        audio,
        (int(audio.shape[0]), 1, int(audio.shape[2]), int(audio.shape[3])),
        strengths[2],
    )
    expanded["noise_mask"] = (
        nested_factory((video_mask, audio_mask))
        if nested_factory is not None
        else (video_mask, audio_mask)
    )
    validate_av_latent(
        expanded,
        timeline_origin_frame=origin,
        name="expanded_av_latent",
    )

    report: dict[str, Any] = {
        "schema": "cauce.h3-av-canvas-expansion-report/1",
        "timeline_origin_frame": origin,
        "frame_count": total_frames,
        "source_canvas": {
            "width": source_width,
            "height": source_height,
            "latent_width": int(video.shape[4]),
            "latent_height": int(video.shape[3]),
        },
        "target_canvas": {
            "width": width,
            "height": height,
            "latent_width": target_w,
            "latent_height": target_h,
        },
        "source_offset": {"x": x, "y": y},
        "video_strength": {"source": strengths[0], "new_region": strengths[1]},
        "audio_strength": strengths[2],
        "mask_digest": _mask_digest(video_mask),
        "requires_comfyui_core": "ff6c8a8af144fc9e9e7bc436b1b202f9316848d8-or-newer",
    }
    report["expansion_hash"] = content_hash(report)
    return expanded, report


def plan_h3_temporal_densification(
    source_frame_count: int,
    factor: int,
) -> dict[str, Any]:
    """Map native H3 visual tokens onto a slower model-time lattice.

    H3 still samples at 24 fps.  Delivering the cropped result at ``24*factor``
    restores the source duration while retaining the frames synthesized in the
    gaps.  Mapping happens on visual-token centres, not decoded frames, so every
    preserved unit is independently maskable by current H3 core.
    """

    source_frames = int(source_frame_count)
    multiplier = int(factor)
    if not is_h3_frame_count(source_frames):
        raise ValueError("source_frame_count must satisfy the H3 17k+5 grid")
    if multiplier < 2 or multiplier > 4:
        raise ValueError("factor must be an integer from 2 through 4")

    delivery_frames = (source_frames - 1) * multiplier + 1
    target_frames = ceil_h3_frame_count(delivery_frames)
    source_token_count = h3_visual_latent_frames(source_frames)
    target_token_count = h3_visual_latent_frames(target_frames)
    source_spans = visual_token_spans(source_token_count)
    target_spans = visual_token_spans(target_token_count)

    def center(span: tuple[int, int]) -> float:
        return (float(span[0]) + float(span[1] - 1)) / 2.0

    anchors: list[dict[str, Any]] = []
    previous = -1
    for source_index, source_span in enumerate(source_spans):
        desired_center = center(source_span) * multiplier
        remaining = source_token_count - source_index - 1
        first = previous + 1
        last = target_token_count - remaining - 1
        candidates = range(first, last + 1)
        target_index = min(
            candidates,
            key=lambda index: (
                abs(center(target_spans[index]) - desired_center),
                -index,
            ),
        )
        target_span = target_spans[target_index]
        anchors.append(
            {
                "source_token": source_index,
                "source_frame_span": list(source_span),
                "target_token": target_index,
                "target_frame_span": list(target_span),
                "desired_target_center": desired_center,
                "resolved_target_center": center(target_span),
            }
        )
        previous = target_index

    anchor_indices = [int(item["target_token"]) for item in anchors]
    anchor_set = set(anchor_indices)
    generated_indices = [index for index in range(target_token_count) if index not in anchor_set]
    return {
        "schema": "cauce.h3-temporal-densification-plan/1",
        "method": "native-token-temporal-inpainting",
        "source_frame_count": source_frames,
        "source_video_tokens": source_token_count,
        "factor": multiplier,
        "h3_model_fps": 24,
        "delivery_fps": 24 * multiplier,
        "delivery_frame_count": delivery_frames,
        "h3_target_frame_count": target_frames,
        "h3_target_video_tokens": target_token_count,
        "decoded_tail_trim_frames": target_frames - delivery_frames,
        "anchors": anchors,
        "anchor_target_tokens": anchor_indices,
        "generated_target_tokens": generated_indices,
        "inside_h3_trained_frame_range": 124 <= target_frames <= 362,
        "duration_seconds": float((source_frames - 1) / 24.0),
        "limitations": [
            "Preservation is exact in packed H3 visual-token state, not pixel-exact after VAE decoding.",
            "H3 interprets the target as slower 24 fps model time; delivery fps restores editorial duration.",
            "The structural-audio stream is regenerated only to satisfy the joint model and should be discarded when the fixed production soundtrack is used.",
        ],
    }


def densify_h3_video_tokens(
    latent: Mapping[str, Any],
    *,
    factor: int,
    anchor_denoise: float = 0.0,
    gap_denoise: float = 1.0,
    feather_tokens: int = 1,
    curve: str = "smootherstep",
    audio_denoise: float = 1.0,
    nested_factory: NestedFactory | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Dilate one H3 visual-token stream and mark inserted tokens for inpainting."""

    video, audio, source_frames = validate_av_latent(latent)
    values = {
        "anchor_denoise": float(anchor_denoise),
        "gap_denoise": float(gap_denoise),
        "audio_denoise": float(audio_denoise),
    }
    if any(value < 0.0 or value > 1.0 for value in values.values()):
        raise ValueError("denoise strengths must lie in [0,1]")
    feather = int(feather_tokens)
    if feather < 1:
        raise ValueError("feather_tokens must be at least one")
    if curve not in {"linear", "smoothstep", "smootherstep"}:
        raise ValueError("curve must be linear, smoothstep, or smootherstep")

    plan = plan_h3_temporal_densification(source_frames, factor)
    target_tokens = int(plan["h3_target_video_tokens"])
    target_frames = int(plan["h3_target_frame_count"])
    target_audio_tokens = h3_audio_token_boundary(target_frames)
    target_video = _new_zeros(
        video,
        (
            int(video.shape[0]),
            int(video.shape[1]),
            target_tokens,
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
            target_audio_tokens,
        ),
    )
    anchor_indices: list[int] = []
    for anchor in plan["anchors"]:
        source_index = int(anchor["source_token"])
        target_index = int(anchor["target_token"])
        target_video[:, :, target_index] = video[:, :, source_index]
        anchor_indices.append(target_index)

    profile: list[float] = []
    for target_index in range(target_tokens):
        distance = min(abs(target_index - anchor) for anchor in anchor_indices)
        weight = _curve(min(1.0, distance / float(feather)), curve)
        profile.append(
            values["anchor_denoise"] + (values["gap_denoise"] - values["anchor_denoise"]) * weight
        )
    video_profile = _profile_tensor(video, profile)
    video_mask = _broadcast_video_mask(video_profile, target_video)
    audio_mask = _new_full(
        audio,
        (int(audio.shape[0]), 1, int(audio.shape[2]), target_audio_tokens),
        values["audio_denoise"],
    )
    out = _with_streams(latent, target_video, target_audio, nested_factory)
    out["noise_mask"] = (
        nested_factory((video_mask, audio_mask))
        if nested_factory is not None
        else (video_mask, audio_mask)
    )
    report = dict(plan)
    report.update(
        {
            "anchor_denoise": values["anchor_denoise"],
            "gap_denoise": values["gap_denoise"],
            "audio_denoise": values["audio_denoise"],
            "feather_tokens": feather,
            "curve": curve,
            "video_mask_sha256": _mask_digest(video_mask),
            "audio_mask_sha256": _mask_digest(audio_mask),
        }
    )
    validate_av_latent(out, name="densified_av_latent")
    return out, report


def resize_h3_av_latent(
    latent: Mapping[str, Any],
    *,
    target_width: int,
    target_height: int,
    method: str = "bicubic",
    video_denoise: float = 1.0,
    audio_denoise: float = 0.0,
    nested_factory: NestedFactory | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Spatially enlarge H3 video state for a same-model second pass."""

    video, audio, frame_count = validate_av_latent(latent)
    width = int(target_width)
    height = int(target_height)
    if width % 32 or height % 32:
        raise ValueError("H3 target width and height must be multiples of 32 pixels")
    if width < int(video.shape[-1]) * 16 or height < int(video.shape[-2]) * 16:
        raise ValueError("H3 spatial regeneration may preserve or enlarge, not shrink")
    strengths = (float(video_denoise), float(audio_denoise))
    if any(value < 0.0 or value > 1.0 for value in strengths):
        raise ValueError("denoise strengths must lie in [0,1]")
    resized_video = _resize_spatial(video, height // 16, width // 16, method)
    kept_audio = _clone(audio)
    video_mask = _new_full(
        video,
        (int(video.shape[0]), 1, int(video.shape[2]), height // 16, width // 16),
        strengths[0],
    )
    audio_mask = _new_full(
        audio,
        (int(audio.shape[0]), 1, int(audio.shape[2]), int(audio.shape[3])),
        strengths[1],
    )
    out = _with_streams(latent, resized_video, kept_audio, nested_factory)
    out["noise_mask"] = (
        nested_factory((video_mask, audio_mask))
        if nested_factory is not None
        else (video_mask, audio_mask)
    )
    validate_av_latent(out, name="resized_av_latent")
    return out, {
        "schema": "cauce.h3-spatial-regeneration-plan/1",
        "method": "latent-hires-second-pass",
        "frame_count": frame_count,
        "source_width": int(video.shape[-1]) * 16,
        "source_height": int(video.shape[-2]) * 16,
        "target_width": width,
        "target_height": height,
        "resize_method": method,
        "video_denoise": strengths[0],
        "audio_denoise": strengths[1],
        "conditioning_rule": "rebuild official H3 conditioning at target geometry",
        "video_mask_sha256": _mask_digest(video_mask),
        "audio_mask_sha256": _mask_digest(audio_mask),
    }


def replace_h3_video_stream(
    target_av_latent: Mapping[str, Any],
    encoded_video_latent: Mapping[str, Any],
    *,
    video_denoise: float = 1.0,
    audio_denoise: float = 0.0,
    nested_factory: NestedFactory | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Graft a VAE-encoded visual stream onto a compatible H3 AV carrier."""

    _old_video, audio, frame_count = validate_av_latent(target_av_latent)
    video = encoded_video_latent.get("samples")
    if video is None or getattr(video, "ndim", None) != 5:
        raise ValueError("encoded_video_latent must contain [B,C,T,H,W] samples")
    if int(video.shape[0]) != int(audio.shape[0]) or int(video.shape[1]) != 24:
        raise ValueError("encoded video batch/channels are incompatible with H3 AV state")
    expected_tokens = h3_visual_latent_frames(frame_count)
    if int(video.shape[2]) != expected_tokens:
        raise ValueError("encoded video duration differs from the H3 AV carrier")
    if int(video.shape[-2]) % 2 or int(video.shape[-1]) % 2:
        raise ValueError("encoded video H/W must align to H3's 2x2 DiT patch grid")
    strengths = (float(video_denoise), float(audio_denoise))
    if any(value < 0.0 or value > 1.0 for value in strengths):
        raise ValueError("denoise strengths must lie in [0,1]")
    video = _clone(video)
    audio = _clone(audio)
    video_mask = _new_full(
        video,
        (int(video.shape[0]), 1, int(video.shape[2]), int(video.shape[3]), int(video.shape[4])),
        strengths[0],
    )
    audio_mask = _new_full(
        audio,
        (int(audio.shape[0]), 1, int(audio.shape[2]), int(audio.shape[3])),
        strengths[1],
    )
    out = _with_streams(target_av_latent, video, audio, nested_factory)
    out["noise_mask"] = (
        nested_factory((video_mask, audio_mask))
        if nested_factory is not None
        else (video_mask, audio_mask)
    )
    validate_av_latent(out, name="grafted_av_latent")
    return out, {
        "schema": "cauce.h3-video-stream-graft/1",
        "method": "pixel-vae-second-pass",
        "frame_count": frame_count,
        "width": int(video.shape[-1]) * 16,
        "height": int(video.shape[-2]) * 16,
        "video_denoise": strengths[0],
        "audio_denoise": strengths[1],
        "conditioning_rule": "rebuild official H3 conditioning at grafted geometry",
        "video_mask_sha256": _mask_digest(video_mask),
        "audio_mask_sha256": _mask_digest(audio_mask),
    }
