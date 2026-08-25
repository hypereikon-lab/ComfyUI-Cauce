"""Validation and shape inspection for packed MiniMax H3 AV latents."""

from __future__ import annotations

from typing import Any

from collections.abc import Mapping

from .timebase import h3_audio_token_boundary, is_h3_frame_count, visual_span_for_tokens


VIDEO_LATENT_CHANNELS = 24
AUDIO_LATENT_CHANNELS = 32
AUDIO_CHANNELS = 2


def get_av_streams(latent: Mapping[str, Any]) -> tuple[Any, Any]:
    """Return the visual and structural-audio streams from an H3 latent."""

    if not isinstance(latent, Mapping):
        raise TypeError("latent must be a mapping containing a samples entry")
    samples = latent.get("samples")
    if samples is None:
        raise ValueError("latent has no samples")
    if hasattr(samples, "tensors"):
        streams = list(samples.tensors)
    elif getattr(samples, "is_nested", False):
        streams = list(samples.unbind())
    elif isinstance(samples, (list, tuple)):
        streams = list(samples)
    else:
        raise ValueError("expected a nested MiniMax H3 audiovisual latent")
    if len(streams) != 2:
        raise ValueError("H3 latent must contain exactly visual and structural-audio streams")
    video, audio = streams[0], streams[1]
    if getattr(video, "ndim", 0) != 5 or getattr(audio, "ndim", 0) != 4:
        raise ValueError("unexpected H3 audiovisual latent shapes")
    return video, audio


def pixel_frames_from_video_latent(video: Any) -> int:
    """Resolve the visible-frame span represented by an H3 visual stream."""

    return visual_span_for_tokens(int(video.shape[2]))


def validate_av_latent(
    latent: Mapping[str, Any],
    *,
    timeline_origin_frame: int = 0,
    name: str = "latent",
) -> tuple[Any, Any, int]:
    """Validate an H3 AV latent/window against one absolute timeline origin."""

    origin = int(timeline_origin_frame)
    if origin < 0:
        raise ValueError("timeline_origin_frame cannot be negative")
    video, audio = get_av_streams(latent)
    if int(video.shape[0]) != 1 or int(audio.shape[0]) != 1:
        raise ValueError(f"{name} video and audio batch sizes must both be 1")
    if int(video.shape[1]) != VIDEO_LATENT_CHANNELS:
        raise ValueError(f"{name} video must have {VIDEO_LATENT_CHANNELS} channels")
    if int(audio.shape[1]) != AUDIO_LATENT_CHANNELS:
        raise ValueError(f"{name} audio must have {AUDIO_LATENT_CHANNELS} latent channels")
    if int(audio.shape[2]) != AUDIO_CHANNELS:
        raise ValueError(f"{name} audio must have {AUDIO_CHANNELS} channels")
    if int(video.shape[3]) < 1 or int(video.shape[4]) < 1:
        raise ValueError(f"{name} video spatial dimensions must be positive")
    if int(audio.shape[-1]) < 1:
        raise ValueError(f"{name} audio temporal length must be positive")

    frame_count = pixel_frames_from_video_latent(video)
    if not is_h3_frame_count(frame_count):
        raise ValueError(f"{name} video temporal length is not a complete 17k+5 H3 run")
    expected_audio = h3_audio_token_boundary(origin + frame_count) - h3_audio_token_boundary(
        origin
    )
    if int(audio.shape[-1]) != expected_audio:
        raise ValueError(
            f"{name} audio temporal length must be {expected_audio} for frames "
            f"[{origin}, {origin + frame_count}), got {int(audio.shape[-1])}"
        )
    return video, audio, frame_count
