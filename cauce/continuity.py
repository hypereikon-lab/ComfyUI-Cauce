"""Latent continuation, acceptance, and exact decoded AV trimming."""

from __future__ import annotations

from typing import Any

from .contracts import WINDOW_SCHEMA
from .h3 import get_av_streams, pixel_frames_from_video_latent
from .timebase import (
    H3_AUDIO_LATENT_HZ,
    H3_FPS,
    frame_to_sample,
    h3_audio_latent_frames,
    h3_av_boundaries,
    h3_visual_latent_frames,
    is_h3_av_boundary,
    is_h3_frame_count,
    visual_token_count_for_span,
)


VALID_CONTEXT_FRAMES = h3_av_boundaries(345)


def _nested_tensor(streams):
    try:
        import comfy.nested_tensor  # type: ignore
    except ImportError as exc:  # pragma: no cover - requires ComfyUI
        raise RuntimeError("H3 continuation can only be built inside ComfyUI") from exc
    return comfy.nested_tensor.NestedTensor(streams)


def extract_av_tail(latent: dict[str, Any], context_frames: int):
    video, audio = get_av_streams(latent)
    context_frames = int(context_frames)
    if not is_h3_av_boundary(context_frames):
        raise ValueError(
            "AV context must end on a shared H3 boundary: 39, 90, 141, 192, ... frames"
        )
    source_frames = pixel_frames_from_video_latent(video)
    if not is_h3_frame_count(source_frames):
        raise ValueError(
            "continuation source is phase-shifted; use a full H3 latent or CAUCE parent latent"
        )
    expected_audio = h3_audio_latent_frames(source_frames)
    if int(audio.shape[-1]) != expected_audio:
        raise ValueError("continuation source video/audio latent clocks disagree")
    tokens = visual_token_count_for_span(context_frames)
    total_tokens = int(video.shape[2])
    if tokens >= total_tokens:
        raise ValueError("context must be shorter than the source video latent")
    start = total_tokens - tokens
    if start % 5 != 0:
        raise ValueError(
            "the requested tail does not begin at H3 token-cycle position zero"
        )
    audio_position = context_frames / H3_FPS * H3_AUDIO_LATENT_HZ
    if audio_position.denominator != 1:
        raise ValueError("context ends between H3 audio latent ticks")
    audio_steps = int(audio_position)
    if audio_steps >= int(audio.shape[-1]):
        raise ValueError("context must be shorter than the source audio latent")
    return (
        video[:, :, start:].clone(),
        audio[..., -audio_steps:].clone(),
    )


def extract_av_head(latent: dict[str, Any], context_frames: int):
    video, audio = get_av_streams(latent)
    context_frames = int(context_frames)
    if not is_h3_av_boundary(context_frames):
        raise ValueError(
            "AV context must end on a shared H3 boundary: 39, 90, 141, 192, ... frames"
        )
    source_frames = pixel_frames_from_video_latent(video)
    if not is_h3_frame_count(source_frames):
        raise ValueError(
            "bridge source is phase-shifted; use a full H3 latent or CAUCE parent latent"
        )
    if int(audio.shape[-1]) != h3_audio_latent_frames(source_frames):
        raise ValueError("bridge source video/audio latent clocks disagree")
    video_steps = visual_token_count_for_span(context_frames)
    audio_steps = int(context_frames / H3_FPS * H3_AUDIO_LATENT_HZ)
    if video_steps >= int(video.shape[2]) or audio_steps >= int(audio.shape[-1]):
        raise ValueError("bridge context must be shorter than its source AV latent")
    return (
        video[:, :, :video_steps].clone(),
        audio[..., :audio_steps].clone(),
    )


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


def _feather_audio_prefix(audio_mask: Any, audio_steps: int, feather_ticks: int):
    import torch

    feather = max(0, min(int(feather_ticks), int(audio_steps)))
    hard = int(audio_steps) - feather
    if hard:
        audio_mask[..., :hard] = 0.0
    if feather:
        index = torch.arange(
            1, feather + 1, device=audio_mask.device, dtype=audio_mask.dtype
        )
        ramp = 0.5 - 0.5 * torch.cos(torch.pi * index / float(feather))
        view_shape = [1] * audio_mask.ndim
        view_shape[-1] = feather
        audio_mask[..., hard:audio_steps] = torch.minimum(
            audio_mask[..., hard:audio_steps], ramp.view(*view_shape)
        )


def _feather_audio_suffix(audio_mask: Any, audio_steps: int, feather_ticks: int):
    import torch

    feather = max(0, min(int(feather_ticks), int(audio_steps)))
    hard = int(audio_steps) - feather
    if hard:
        audio_mask[..., -hard:] = 0.0
    if feather:
        index = torch.arange(
            feather, 0, -1, device=audio_mask.device, dtype=audio_mask.dtype
        )
        ramp = 0.5 - 0.5 * torch.cos(torch.pi * index / float(feather))
        view_shape = [1] * audio_mask.ndim
        view_shape[-1] = feather
        start = int(audio_mask.shape[-1]) - int(audio_steps)
        audio_mask[..., start : start + feather] = torch.minimum(
            audio_mask[..., start : start + feather], ramp.view(*view_shape)
        )


def prepare_continuation(
    positive: Any,
    target_latent: dict[str, Any],
    previous_latent: dict[str, Any],
    *,
    context_frames: int = 39,
    audio_feather_ticks: int = 8,
    conditioning_mode: str = "mask_only",
) -> tuple[Any, dict[str, Any], int]:
    """Pin a previous AV latent tail into the head of a new H3 window."""

    context_frames = int(context_frames)
    if context_frames not in VALID_CONTEXT_FRAMES:
        raise ValueError(f"context_frames must be one of {VALID_CONTEXT_FRAMES}")
    if conditioning_mode not in {"mask_only", "mask_plus_guide"}:
        raise ValueError("conditioning_mode must be mask_only or mask_plus_guide")
    target_video, target_audio = get_av_streams(target_latent)
    tail_video, tail_audio = extract_av_tail(previous_latent, context_frames)
    if int(target_video.shape[0]) != int(tail_video.shape[0]) or int(
        target_audio.shape[0]
    ) != int(tail_audio.shape[0]):
        raise ValueError("previous and target H3 AV latent batch sizes differ")
    if tuple(target_video.shape[1:2] + target_video.shape[3:]) != tuple(
        tail_video.shape[1:2] + tail_video.shape[3:]
    ):
        raise ValueError("previous and target H3 video latents have different geometry")
    if tuple(target_audio.shape[1:3]) != tuple(tail_audio.shape[1:3]):
        raise ValueError("previous and target H3 audio latents have different geometry")
    if int(tail_video.shape[2]) >= int(target_video.shape[2]):
        raise ValueError("target window is too short for the requested video context")
    if int(tail_audio.shape[-1]) >= int(target_audio.shape[-1]):
        raise ValueError("target window is too short for the requested audio context")

    video = target_video.clone()
    audio = target_audio.clone()
    video[:, :, : tail_video.shape[2]] = tail_video.to(video)
    audio[..., : tail_audio.shape[-1]] = tail_audio.to(audio)

    video_mask, audio_mask = _base_masks(target_latent, video, audio)
    video_mask[:, :, : tail_video.shape[2]] = 0.0
    audio_steps = int(tail_audio.shape[-1])
    _feather_audio_prefix(audio_mask, audio_steps, audio_feather_ticks)

    conditioned = positive
    if conditioning_mode == "mask_plus_guide":
        try:
            import node_helpers  # type: ignore
        except ImportError as exc:  # pragma: no cover - requires ComfyUI
            raise RuntimeError("H3 continuation can only be built inside ComfyUI") from exc
        keyframe = {
            "resolved_frame_index": 0,
            "latent": tail_video,
            "audio_latent": tail_audio,
        }
        conditioned = node_helpers.conditioning_set_values(
            positive, {"minimax_keyframes": [keyframe]}, append=True
        )
    out = dict(target_latent)
    out["samples"] = _nested_tensor((video, audio))
    out["noise_mask"] = _nested_tensor((video_mask, audio_mask))
    return conditioned, out, context_frames


def prepare_bridge(
    positive: Any,
    target_latent: dict[str, Any],
    left_parent: dict[str, Any],
    right_parent: dict[str, Any],
    *,
    context_frames: int = 39,
    audio_feather_ticks: int = 8,
    conditioning_mode: str = "mask_only",
) -> tuple[Any, dict[str, Any], int]:
    """Protect phase-aligned AV endpoints and generate only their middle."""

    context_frames = int(context_frames)
    if context_frames not in VALID_CONTEXT_FRAMES:
        raise ValueError(f"context_frames must be one of {VALID_CONTEXT_FRAMES}")
    if conditioning_mode not in {"mask_only", "mask_plus_guide"}:
        raise ValueError("conditioning_mode must be mask_only or mask_plus_guide")

    target_video, target_audio = get_av_streams(target_latent)
    left_video, left_audio = extract_av_tail(left_parent, context_frames)
    right_video, right_audio = extract_av_head(right_parent, context_frames)
    if len(
        {
            int(target_video.shape[0]),
            int(target_audio.shape[0]),
            int(left_video.shape[0]),
            int(left_audio.shape[0]),
            int(right_video.shape[0]),
            int(right_audio.shape[0]),
        }
    ) != 1:
        raise ValueError("bridge parent and target H3 AV latent batch sizes differ")
    if tuple(left_video.shape[1:2] + left_video.shape[3:]) != tuple(
        target_video.shape[1:2] + target_video.shape[3:]
    ) or tuple(right_video.shape[1:2] + right_video.shape[3:]) != tuple(
        target_video.shape[1:2] + target_video.shape[3:]
    ):
        raise ValueError("bridge parent and target video latent geometry differs")
    if tuple(left_audio.shape[1:3]) != tuple(target_audio.shape[1:3]) or tuple(
        right_audio.shape[1:3]
    ) != tuple(target_audio.shape[1:3]):
        raise ValueError("bridge parent and target audio latent geometry differs")
    video_steps = int(left_video.shape[2])
    audio_steps = int(left_audio.shape[-1])
    if int(right_video.shape[2]) != video_steps or int(right_audio.shape[-1]) != audio_steps:
        raise ValueError("bridge endpoints resolved to different AV context lengths")
    if video_steps * 2 >= int(target_video.shape[2]):
        raise ValueError("bridge video contexts leave no generable latent middle")
    if audio_steps * 2 >= int(target_audio.shape[-1]):
        raise ValueError("bridge audio contexts leave no generable latent middle")

    video = target_video.clone()
    audio = target_audio.clone()
    video[:, :, :video_steps] = left_video.to(video)
    video[:, :, -video_steps:] = right_video.to(video)
    audio[..., :audio_steps] = left_audio.to(audio)
    audio[..., -audio_steps:] = right_audio.to(audio)
    video_mask, audio_mask = _base_masks(target_latent, video, audio)
    video_mask[:, :, :video_steps] = 0.0
    video_mask[:, :, -video_steps:] = 0.0
    _feather_audio_prefix(audio_mask, audio_steps, audio_feather_ticks)
    _feather_audio_suffix(audio_mask, audio_steps, audio_feather_ticks)

    conditioned = positive
    if conditioning_mode == "mask_plus_guide":
        try:
            import node_helpers  # type: ignore
        except ImportError as exc:  # pragma: no cover - requires ComfyUI
            raise RuntimeError("H3 bridge can only be built inside ComfyUI") from exc
        target_frames = pixel_frames_from_video_latent(target_video)
        conditioned = node_helpers.conditioning_set_values(
            positive,
            {
                "minimax_keyframes": [
                    {
                        "resolved_frame_index": 0,
                        "latent": left_video,
                        "audio_latent": left_audio,
                    },
                    {
                        "resolved_frame_index": target_frames - context_frames,
                        "latent": right_video,
                        "audio_latent": right_audio,
                    },
                ]
            },
            append=True,
        )
    out = dict(target_latent)
    out["samples"] = _nested_tensor((video, audio))
    out["noise_mask"] = _nested_tensor((video_mask, audio_mask))
    middle_frames = pixel_frames_from_video_latent(target_video) - 2 * context_frames
    return conditioned, out, middle_frames


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
