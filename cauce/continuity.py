"""Visual latent continuation, acceptance, and exact decoded trimming."""

from __future__ import annotations

from typing import Any

from .contracts import WINDOW_SCHEMA
from .h3 import get_av_streams, pixel_frames_from_video_latent
from .timebase import (
    frame_to_sample,
    h3_audio_latent_frames,
    h3_visual_latent_frames,
    is_h3_frame_count,
    visual_token_count_for_span,
)


VALID_CONTEXT_FRAMES = tuple(range(5, 346, 17))


def _nested_tensor(streams):
    try:
        import comfy.nested_tensor  # type: ignore
    except ImportError as exc:  # pragma: no cover - requires ComfyUI
        raise RuntimeError("H3 continuation can only be built inside ComfyUI") from exc
    return comfy.nested_tensor.NestedTensor(streams)


def extract_video_tail(latent: dict[str, Any], context_frames: int):
    video, _ = get_av_streams(latent)
    context_frames = int(context_frames)
    if not is_h3_frame_count(context_frames):
        raise ValueError(
            "visual context must use the H3 grid: 5, 22, 39, 56, ... frames"
        )
    source_frames = pixel_frames_from_video_latent(video)
    if not is_h3_frame_count(source_frames):
        raise ValueError(
            "continuation source is phase-shifted; use a full H3 latent or CAUCE parent latent"
        )
    tokens = visual_token_count_for_span(context_frames)
    total_tokens = int(video.shape[2])
    if tokens >= total_tokens:
        raise ValueError("context must be shorter than the source video latent")
    start = total_tokens - tokens
    if start % 5 != 0:
        raise ValueError(
            "the requested tail does not begin at H3 token-cycle position zero"
        )
    return video[:, :, start:].clone()


def _base_masks(target_latent: dict[str, Any], video: Any, audio: Any):
    import torch

    prior_mask = target_latent.get("noise_mask")
    if prior_mask is not None:
        if not getattr(prior_mask, "is_nested", False):
            raise ValueError("an H3 target must use a nested AV noise mask")
        prior_streams = list(prior_mask.unbind())
        if len(prior_streams) < 2:
            raise ValueError("target AV noise mask is missing its audio stream")
        return (
            prior_streams[0].to(device=video.device, dtype=torch.float32).clone(),
            prior_streams[1].to(device=audio.device, dtype=torch.float32).clone(),
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
    positive: Any,
    target_latent: dict[str, Any],
    previous_latent: dict[str, Any],
    *,
    context_frames: int = 39,
    conditioning_mode: str = "mask_only",
) -> tuple[Any, dict[str, Any], int]:
    """Pin a previous visual latent tail while freezing H3's audio scaffolding."""

    context_frames = int(context_frames)
    if context_frames not in VALID_CONTEXT_FRAMES:
        raise ValueError(f"context_frames must be one of {VALID_CONTEXT_FRAMES}")
    if conditioning_mode not in {"mask_only", "mask_plus_guide"}:
        raise ValueError("conditioning_mode must be mask_only or mask_plus_guide")
    target_video, target_audio = get_av_streams(target_latent)
    tail_video = extract_video_tail(previous_latent, context_frames)
    if int(target_video.shape[0]) != int(tail_video.shape[0]):
        raise ValueError("previous and target H3 video latent batch sizes differ")
    if tuple(target_video.shape[1:2] + target_video.shape[3:]) != tuple(
        tail_video.shape[1:2] + tail_video.shape[3:]
    ):
        raise ValueError("previous and target H3 video latents have different geometry")
    if int(tail_video.shape[2]) >= int(target_video.shape[2]):
        raise ValueError("target window is too short for the requested video context")

    video = target_video.clone()
    audio = target_audio.clone()
    video[:, :, : tail_video.shape[2]] = tail_video.to(video)

    video_mask, audio_mask = _base_masks(target_latent, video, audio)
    video_mask[:, :, : tail_video.shape[2]] = 0.0
    audio_mask.zero_()

    conditioned = positive
    if conditioning_mode == "mask_plus_guide":
        from .h3 import official_h3_nodes

        _, _, add_guide = official_h3_nodes()
        if add_guide is None:
            raise RuntimeError(
                "mask_plus_guide needs a ComfyUI H3 runtime with the official "
                "MiniMaxH3AddGuide clip-keyframe implementation; use mask_only or "
                "the decoded-endpoint continuation workflow on this runtime"
            )
        try:
            import node_helpers  # type: ignore
        except ImportError as exc:  # pragma: no cover - requires ComfyUI
            raise RuntimeError("H3 continuation can only be built inside ComfyUI") from exc
        keyframe = {
            "resolved_frame_index": 0,
            "latent": tail_video,
        }
        keyframes = list(positive[0][1].get("minimax_keyframes", []))
        keyframes.append(keyframe)
        conditioned = node_helpers.conditioning_set_values(
            positive, {"minimax_keyframes": keyframes}
        )
    out = dict(target_latent)
    out["samples"] = _nested_tensor((video, audio))
    out["noise_mask"] = _nested_tensor((video_mask, audio_mask))
    return conditioned, out, context_frames


def resolve_parent_latent(
    latent: dict[str, Any], window: dict[str, Any]
) -> dict[str, Any]:
    """Crop only the post-accept tail, retaining a phase-safe H3 parent origin."""

    if window.get("schema") != WINDOW_SCHEMA:
        raise ValueError(f"window schema must be {WINDOW_SCHEMA}")
    end_frame = int(window["accepted_end_frame"])
    if not is_h3_frame_count(end_frame):
        raise ValueError(
            "accepted endpoint is not a phase-safe H3 run; use nearest_run, floor_run, ceil_run, or full_render"
        )
    video, audio = get_av_streams(latent)
    expected_frames = int(window["shape"]["pixel_frames"])
    if pixel_frames_from_video_latent(video) != expected_frames:
        raise ValueError("window and sampled video latent lengths disagree")
    end_video = h3_visual_latent_frames(end_frame)
    end_audio = h3_audio_latent_frames(end_frame)
    if end_video > int(video.shape[2]) or end_audio > int(audio.shape[-1]):
        raise ValueError("sampled AV latent is shorter than the resolved parent range")
    out = dict(latent)
    out.pop("noise_mask", None)
    out["samples"] = _nested_tensor(
        (
            video[:, :, :end_video].clone(),
            audio[..., :end_audio].clone(),
        )
    )
    return out


def trim_decoded_window(images: Any, trim_frames: int, audio: dict[str, Any] | None = None):
    """Trim image and audio heads, then force audio to the exact frame duration."""

    import torch.nn.functional as functional

    trim_frames = max(0, int(trim_frames))
    total_frames = int(images.shape[0])
    if trim_frames >= total_frames:
        raise ValueError("trim would remove every decoded video frame")
    result_images = images[trim_frames:] if trim_frames else images
    if audio is None:
        return result_images, None

    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 0))
    if waveform is None or sample_rate <= 0:
        raise ValueError("audio must contain waveform and sample_rate")
    cut = frame_to_sample(trim_frames, sample_rate)
    if cut >= int(waveform.shape[-1]):
        raise ValueError("audio is shorter than the requested head trim")
    waveform = waveform[..., cut:]
    expected = frame_to_sample(total_frames - trim_frames, sample_rate)
    have = int(waveform.shape[-1])
    if have > expected:
        waveform = waveform[..., :expected]
    elif have < expected:
        waveform = functional.pad(waveform, (0, expected - have))
    return result_images, {"waveform": waveform, "sample_rate": sample_rate}


def accept_decoded_window(
    images: Any,
    window: dict[str, Any],
    audio: dict[str, Any] | None = None,
):
    """Accept the exact visible-frame range declared by a CAUCE window."""

    import torch.nn.functional as functional

    if window.get("schema") != WINDOW_SCHEMA:
        raise ValueError(f"window schema must be {WINDOW_SCHEMA}")
    start_frame = int(window["accepted_start_frame"])
    end_frame = int(window["accepted_end_frame"])
    total_frames = int(images.shape[0])
    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError("window contains an invalid accepted frame range")
    if end_frame > total_frames:
        raise ValueError(
            f"decoded video has {total_frames} frames but acceptance ends at frame {end_frame}"
        )
    result_images = images[start_frame:end_frame]
    accepted_frames = end_frame - start_frame
    if audio is None:
        return result_images, None, accepted_frames

    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 0))
    if waveform is None or sample_rate <= 0:
        raise ValueError("audio must contain waveform and sample_rate")
    start_sample = frame_to_sample(start_frame, sample_rate)
    end_sample = frame_to_sample(end_frame, sample_rate)
    expected = end_sample - start_sample
    if start_sample >= int(waveform.shape[-1]):
        raise ValueError("audio is shorter than the accepted range start")
    result_waveform = waveform[..., start_sample:min(end_sample, waveform.shape[-1])]
    have = int(result_waveform.shape[-1])
    if have < expected:
        result_waveform = functional.pad(result_waveform, (0, expected - have))
    return (
        result_images,
        {"waveform": result_waveform, "sample_rate": sample_rate},
        accepted_frames,
    )
