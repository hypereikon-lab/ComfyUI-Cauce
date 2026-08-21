"""Compile rational CAUCE fields into native nested H3 denoise masks."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from .contracts import FIELD_SCHEMA, WINDOW_SCHEMA, range_fraction
from .h3 import get_av_streams
from .timebase import (
    H3_AUDIO_LATENT_HZ,
    H3_FPS,
    audio_latent_spans,
    fraction_from_payload,
    reduce_intervals,
    visual_token_spans,
)


def _field_sources(field: dict[str, Any] | None, channel: str):
    if field is None:
        return ()
    if field.get("schema") != FIELD_SCHEMA:
        raise ValueError(f"field schema must be {FIELD_SCHEMA}")
    sources = []
    for item in field.get("spans", []):
        if item.get("channel") not in {channel, "both"}:
            continue
        start, end = range_fraction(item["range"])
        sources.append((start, end, float(item["strength"])))
    return tuple(sources)


def compile_video_field(
    window: dict[str, Any],
    field: dict[str, Any] | None,
    *,
    default: float = 1.0,
) -> tuple[float, ...]:
    if window.get("schema") != WINDOW_SCHEMA:
        raise ValueError(f"window schema must be {WINDOW_SCHEMA}")
    render_start, _ = range_fraction(window["render_range"])
    token_count = int(window["shape"]["video_latent_frames"])
    spans = tuple(
        (
            render_start + Fraction(start, int(H3_FPS)),
            render_start + Fraction(end, int(H3_FPS)),
        )
        for start, end in visual_token_spans(token_count)
    )
    return reduce_intervals(spans, _field_sources(field, "video"), default)


def compile_audio_field(
    window: dict[str, Any],
    field: dict[str, Any] | None,
    *,
    default: float = 1.0,
) -> tuple[float, ...]:
    if window.get("schema") != WINDOW_SCHEMA:
        raise ValueError(f"window schema must be {WINDOW_SCHEMA}")
    render_start, _ = range_fraction(window["render_range"])
    count = int(window["shape"]["audio_latent_frames"])
    spans = tuple(
        (render_start + start, render_start + end)
        for start, end in audio_latent_spans(count)
    )
    return reduce_intervals(spans, _field_sources(field, "audio"), default)


def _resize_mask(mask: Any, height: int, width: int):
    import torch.nn.functional as functional

    return functional.interpolate(
        mask.unsqueeze(1).to(dtype=mask.dtype),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[:, 0]


def reduce_pixel_mask_to_h3(
    pixel_mask: Any,
    *,
    pixel_frames: int,
    video_latent_frames: int,
    latent_height: int,
    latent_width: int,
):
    """Reduce a visible-frame mask over every frame covered by each H3 token."""

    import torch

    mask = pixel_mask
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 3:
        raise ValueError("pixel mask must have shape [frames,height,width]")
    if int(mask.shape[0]) not in {1, int(pixel_frames)}:
        raise ValueError(
            f"pixel mask has {mask.shape[0]} frames; expected 1 or {pixel_frames}"
        )
    mask = mask.to(dtype=torch.float32).clamp(0.0, 1.0)
    mask = _resize_mask(mask, int(latent_height), int(latent_width))
    if int(mask.shape[0]) == 1:
        mask = mask.expand(int(pixel_frames), -1, -1)

    reduced = []
    spans = visual_token_spans(int(video_latent_frames))
    if spans[-1][1] != int(pixel_frames):
        raise ValueError("window frame count and H3 latent token count disagree")
    for start, end in spans:
        reduced.append(mask[start:end].amax(dim=0))
    return torch.stack(reduced, dim=0).unsqueeze(0).unsqueeze(0)


def compile_nested_av_mask(
    latent: dict[str, Any],
    window: dict[str, Any],
    *,
    field: dict[str, Any] | None = None,
    spatial_mask: Any = None,
    default_video: float = 1.0,
    default_audio: float = 1.0,
):
    """Return a Comfy NestedTensor mask with 1=generate and 0=preserve."""

    import torch

    try:
        import comfy.nested_tensor  # type: ignore
    except ImportError as exc:  # pragma: no cover - requires ComfyUI
        raise RuntimeError("nested H3 masks can only be built inside ComfyUI") from exc

    video, audio = get_av_streams(latent)
    shape = window["shape"]
    if int(video.shape[2]) != int(shape["video_latent_frames"]):
        raise ValueError("window and video latent lengths disagree")
    if int(audio.shape[-1]) != int(shape["audio_latent_frames"]):
        raise ValueError("window and audio latent lengths disagree")

    video_values = torch.tensor(
        compile_video_field(window, field, default=default_video),
        dtype=torch.float32,
        device=video.device,
    ).view(1, 1, int(video.shape[2]), 1, 1)
    video_mask = video_values.expand(
        int(video.shape[0]), 1, int(video.shape[2]), int(video.shape[3]), int(video.shape[4])
    ).clone()
    if spatial_mask is not None:
        spatial = reduce_pixel_mask_to_h3(
            spatial_mask,
            pixel_frames=int(shape["pixel_frames"]),
            video_latent_frames=int(shape["video_latent_frames"]),
            latent_height=int(video.shape[3]),
            latent_width=int(video.shape[4]),
        ).to(device=video.device)
        video_mask.mul_(spatial)

    audio_values = torch.tensor(
        compile_audio_field(window, field, default=default_audio),
        dtype=torch.float32,
        device=audio.device,
    ).view(1, 1, 1, int(audio.shape[-1]))
    audio_mask = audio_values.expand(
        int(audio.shape[0]), 1, int(audio.shape[2]), int(audio.shape[-1])
    ).clone()
    return comfy.nested_tensor.NestedTensor((video_mask, audio_mask))


def combine_nested_masks(left: Any, right: Any):
    """Intersect two H3 masks; preservation wins over generation."""

    import torch

    if not getattr(left, "is_nested", False) or not getattr(right, "is_nested", False):
        raise ValueError("combining H3 AV masks requires two nested masks")
    left_streams = list(left.unbind())
    right_streams = list(right.unbind())
    if len(left_streams) != len(right_streams):
        raise ValueError("nested AV masks have different stream counts")
    combined = []
    for a, b in zip(left_streams, right_streams):
        try:
            combined.append(torch.minimum(a.to(torch.float32), b.to(a.device, torch.float32)))
        except RuntimeError as exc:
            raise ValueError("nested AV mask shapes are incompatible") from exc
    try:
        import comfy.nested_tensor  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("nested H3 masks can only be combined inside ComfyUI") from exc
    return comfy.nested_tensor.NestedTensor(combined)


def mask_report(
    window: dict[str, Any],
    field: dict[str, Any] | None,
    *,
    default_video: float = 1.0,
    default_audio: float = 1.0,
) -> dict[str, Any]:
    video = compile_video_field(window, field, default=default_video)
    audio = compile_audio_field(window, field, default=default_audio)
    return {
        "schema": "cauce.mask-report/1",
        "video_tokens": len(video),
        "audio_tokens": len(audio),
        "video_generate_mean": sum(video) / len(video),
        "audio_generate_mean": sum(audio) / len(audio),
        "video_preserved_tokens": sum(value < 1.0 - 1e-6 for value in video),
        "audio_preserved_tokens": sum(value < 1.0 - 1e-6 for value in audio),
    }
