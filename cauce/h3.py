"""Minimal validation for persisted H3 audiovisual latents."""

from __future__ import annotations

from typing import Any

from .timebase import visual_span_for_tokens


def get_av_streams(latent: dict[str, Any]) -> tuple[Any, Any]:
    """Return the visual and structural-audio streams from an H3 latent."""

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
        raise ValueError("H3 latent must contain visual and structural-audio streams")
    video, audio = streams[0], streams[1]
    if getattr(video, "ndim", 0) != 5 or getattr(audio, "ndim", 0) != 4:
        raise ValueError("unexpected H3 audiovisual latent shapes")
    return video, audio


def pixel_frames_from_video_latent(video: Any) -> int:
    """Resolve the visible-frame span represented by an H3 visual stream."""

    return visual_span_for_tokens(int(video.shape[2]))
