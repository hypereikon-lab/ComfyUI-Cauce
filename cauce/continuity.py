"""Phase-aware H3 continuation and decoded-range acceptance."""

from __future__ import annotations

from typing import Any

from .h3 import get_av_streams, pixel_frames_from_video_latent
from .timebase import (
    h3_audio_latent_frames,
    h3_visual_latent_frames,
    is_h3_frame_count,
    visual_token_count_for_span,
)


VALID_CONTEXT_FRAMES = tuple(range(5, 346, 17))


def _nested_tensor(streams: tuple[Any, Any]):
    try:
        import comfy.nested_tensor  # type: ignore
    except ImportError as exc:  # pragma: no cover - requires ComfyUI
        raise RuntimeError("H3 continuation requires ComfyUI") from exc
    return comfy.nested_tensor.NestedTensor(streams)


def extract_video_tail(latent: dict[str, Any], context_frames: int):
    """Extract a phase-aligned visual tail from a complete H3 latent."""

    video, _ = get_av_streams(latent)
    context_frames = int(context_frames)
    if not is_h3_frame_count(context_frames):
        raise ValueError("visual context must use the H3 grid: 5, 22, 39, 56, ...")
    source_frames = pixel_frames_from_video_latent(video)
    if not is_h3_frame_count(source_frames):
        raise ValueError("continuation source is not a complete phase-aligned H3 latent")
    tokens = visual_token_count_for_span(context_frames)
    total_tokens = int(video.shape[2])
    if tokens >= total_tokens:
        raise ValueError("context must be shorter than the source visual stream")
    start = total_tokens - tokens
    if start % 5 != 0:
        raise ValueError("the requested tail does not begin at H3 token-cycle position zero")
    return video[:, :, start:].clone()


def _base_masks(target_latent: dict[str, Any], video: Any, audio: Any):
    import torch

    prior_mask = target_latent.get("noise_mask")
    if prior_mask is not None:
        if not getattr(prior_mask, "is_nested", False):
            raise ValueError("an H3 target must use a nested audiovisual noise mask")
        streams = list(prior_mask.unbind())
        if len(streams) < 2:
            raise ValueError("target audiovisual noise mask is incomplete")
        return (
            streams[0].to(device=video.device, dtype=torch.float32).clone(),
            streams[1].to(device=audio.device, dtype=torch.float32).clone(),
        )
    return (
        torch.ones(
            (video.shape[0], 1, video.shape[2], video.shape[3], video.shape[4]),
            dtype=torch.float32,
            device=video.device,
        ),
        torch.ones(
            (audio.shape[0], 1, audio.shape[2], audio.shape[-1]),
            dtype=torch.float32,
            device=audio.device,
        ),
    )


def prepare_continuation(
    target_latent: dict[str, Any],
    previous_latent: dict[str, Any],
    *,
    context_frames: int = 39,
) -> tuple[dict[str, Any], int]:
    """Pin a previous visual tail and freeze H3's structural-audio stream."""

    context_frames = int(context_frames)
    if context_frames not in VALID_CONTEXT_FRAMES:
        raise ValueError(f"context_frames must be one of {VALID_CONTEXT_FRAMES}")
    target_video, target_audio = get_av_streams(target_latent)
    tail_video = extract_video_tail(previous_latent, context_frames)
    if int(target_video.shape[0]) != int(tail_video.shape[0]):
        raise ValueError("source and target H3 batch sizes differ")
    if tuple(target_video.shape[1:2] + target_video.shape[3:]) != tuple(
        tail_video.shape[1:2] + tail_video.shape[3:]
    ):
        raise ValueError("source and target H3 visual streams have different geometry")
    if int(tail_video.shape[2]) >= int(target_video.shape[2]):
        raise ValueError("target latent is too short for the requested context")

    video = target_video.clone()
    audio = target_audio.clone()
    video[:, :, : tail_video.shape[2]] = tail_video.to(video)
    video_mask, audio_mask = _base_masks(target_latent, video, audio)
    video_mask[:, :, : tail_video.shape[2]] = 0.0
    audio_mask.zero_()

    out = dict(target_latent)
    out["samples"] = _nested_tensor((video, audio))
    out["noise_mask"] = _nested_tensor((video_mask, audio_mask))
    return out, context_frames


def resolve_parent_latent(
    latent: dict[str, Any], accepted_end_frame: int
) -> dict[str, Any]:
    """Crop an H3 latent at a phase-safe visible-frame endpoint."""

    end_frame = int(accepted_end_frame)
    if not is_h3_frame_count(end_frame):
        raise ValueError("accepted_end_frame must use the H3 grid: 5, 22, 39, 56, ...")
    video, audio = get_av_streams(latent)
    total_frames = pixel_frames_from_video_latent(video)
    if end_frame > total_frames:
        raise ValueError("accepted endpoint extends beyond the sampled latent")
    end_video = h3_visual_latent_frames(end_frame)
    end_audio = h3_audio_latent_frames(end_frame)
    if end_audio > int(audio.shape[-1]):
        raise ValueError("structural-audio stream is shorter than the accepted endpoint")
    out = dict(latent)
    out.pop("noise_mask", None)
    out["samples"] = _nested_tensor(
        (video[:, :, :end_video].clone(), audio[..., :end_audio].clone())
    )
    return out


def accept_decoded_range(images: Any, start_frame: int, frame_count: int):
    """Return an exact visible-frame slice from a decoded batch."""

    start = int(start_frame)
    count = int(frame_count)
    end = start + count
    total = int(images.shape[0])
    if start < 0 or count < 1:
        raise ValueError("decoded range requires start_frame >= 0 and frame_count >= 1")
    if end > total:
        raise ValueError(f"decoded batch has {total} frames but the range ends at {end}")
    return images[start:end], count
