"""Localized, duration-preserving temporal inpainting for opaque video batches."""

from __future__ import annotations

import copy
import math
from typing import Any

from .contracts import SEAM_SCHEMA, content_hash, make_window
from .h3 import (
    execute_add_guide,
    get_av_streams,
    pixel_frames_from_video_latent,
    require_h3_temporal_edit_runtime,
)
from .timebase import (
    H3_FPS,
    H3_TRAINED_MIN_FRAMES,
    frames_to_seconds,
    h3_visual_latent_frames,
    is_h3_frame_count,
    seconds_to_frames,
    visual_span_for_tokens,
    visual_token_count_for_span,
    visual_token_spans,
)


def _concat_image_batches(parts: list[Any]):
    parts = [part for part in parts if int(part.shape[0]) > 0]
    if not parts:
        raise ValueError("cannot concatenate an empty set of image batches")
    try:
        import torch

        if isinstance(parts[0], torch.Tensor):
            return torch.cat(parts, dim=0)
    except ImportError:  # pragma: no cover - torch ships with ComfyUI
        pass
    import numpy

    return numpy.concatenate(parts, axis=0)


MASK_CURVES = ("cosine", "smoothstep", "linear")
TOKEN_PROJECTIONS = ("cover", "majority")
NATIVE_SEAM_CONTEXT_FRAMES = (22, 39, 56, 73)
NATIVE_SEAM_WORKING_FRAMES = tuple(range(124, 363, 34))


def _curve_value(value: float, curve: str) -> float:
    value = min(1.0, max(0.0, float(value)))
    if curve == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * value)
    if curve == "smoothstep":
        return value * value * (3.0 - 2.0 * value)
    if curve == "linear":
        return value
    raise ValueError(f"mask curve must be one of {', '.join(MASK_CURVES)}")


def _symmetric_weights(length: int, transition_frames: int, curve: str) -> tuple[float, ...]:
    """Return a symmetric opacity field with exact zero-valued endpoints."""

    length = int(length)
    transition = min(max(0, int(transition_frames)), max(0, length // 2))
    if length < 1:
        return ()
    if transition == 0:
        return (1.0,) * length
    values = []
    for index in range(length):
        distance = min(index, length - 1 - index)
        values.append(_curve_value(min(1.0, distance / float(transition)), curve))
    return tuple(values)


def _resize_visible_mask(mask: Any, frames: int, height: int, width: int):
    """Normalize a Comfy MASK to [frames,height,width] without binarizing it."""

    import torch
    import torch.nn.functional as functional

    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4 and int(mask.shape[1]) == 1:
        mask = mask[:, 0]
    if mask.ndim != 3:
        raise ValueError("generation/blend mask must have shape [frames,height,width]")
    if int(mask.shape[0]) not in {1, int(frames)}:
        raise ValueError(
            f"generation/blend mask has {mask.shape[0]} frames; expected 1 or {frames}"
        )
    mask = mask.to(dtype=torch.float32).clamp(0.0, 1.0)
    if tuple(mask.shape[-2:]) != (int(height), int(width)):
        mask = functional.interpolate(
            mask.unsqueeze(1),
            size=(int(height), int(width)),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    if int(mask.shape[0]) == 1:
        mask = mask.expand(int(frames), -1, -1)
    return mask


def _blend_patch(
    patch: Any,
    original: Any,
    feather_frames: int,
    curve: str,
    blend_strength: Any = None,
):
    weights = _symmetric_weights(int(patch.shape[0]), feather_frames, curve)
    try:
        import torch

        if isinstance(patch, torch.Tensor):
            if blend_strength is None:
                alpha = torch.tensor(weights, dtype=patch.dtype, device=patch.device)
                alpha = alpha.view((-1,) + (1,) * (patch.ndim - 1))
            else:
                alpha = _resize_visible_mask(
                    blend_strength,
                    int(patch.shape[0]),
                    int(patch.shape[1]),
                    int(patch.shape[2]),
                ).to(device=patch.device, dtype=patch.dtype)
                alpha = alpha.unsqueeze(-1)
            return original.to(patch) * (1.0 - alpha) + patch * alpha
    except ImportError:  # pragma: no cover
        pass
    import numpy

    alpha = numpy.asarray(weights, dtype=patch.dtype).reshape(
        (-1,) + (1,) * (patch.ndim - 1)
    )
    return original.astype(patch.dtype) * (1.0 - alpha) + patch * alpha


def make_seam_plan(
    left_frame_count: int,
    right_frame_count: int,
    *,
    context_seconds_per_side: float = 2.5,
    repair_seconds_total: float = 3.0,
    guide_frames: int = 22,
    maximum_frames: int = 362,
) -> dict[str, Any]:
    """Resolve one cut into a token-aligned temporal inpaint interval.

    The requested repair duration is the *entire* interval across the cut, not
    a duration per side.  The interval is snapped to symmetric H3 token
    boundaries and surrounded by valid multi-frame guide clips. This keeps the
    generated interval explicit while exposing incoming and outgoing motion.
    """

    left_count = int(left_frame_count)
    right_count = int(right_frame_count)
    context = seconds_to_frames(context_seconds_per_side, H3_FPS, "nearest")
    requested_repair = seconds_to_frames(repair_seconds_total, H3_FPS, "nearest")
    guide_frames = int(guide_frames)
    if context < 1 or requested_repair < 1:
        raise ValueError("context and repair durations must resolve to at least one frame")
    if not is_h3_frame_count(guide_frames):
        raise ValueError("guide_frames must use the H3 clip grid: 5, 22, 39, 56, ...")
    if left_count < context or right_count < context:
        raise ValueError(
            "each source needs at least "
            f"{context} frames ({float(frames_to_seconds(context)):.3f}s)"
        )

    real_frames = context * 2
    working_frames = None
    for candidate in range(
        max(real_frames, H3_TRAINED_MIN_FRAMES), int(maximum_frames) + 1
    ):
        if is_h3_frame_count(candidate) and (candidate - real_frames) % 2 == 0:
            working_frames = candidate
            break
    if working_frames is None:
        raise ValueError(
            "no symmetric H3 guard-frame solution fits the requested context and maximum"
        )
    guard = (working_frames - real_frames) // 2
    cut = guard + context
    boundaries = {0}
    boundaries.update(
        end
        for _, end in visual_token_spans(h3_visual_latent_frames(working_frames))
    )
    candidates = []
    for start in boundaries:
        end = 2 * cut - start
        if (
            start < cut < end
            and end in boundaries
            and start - guide_frames >= 0
            and end + guide_frames <= working_frames
        ):
            length = end - start
            candidates.append((start, end, length))
    if not candidates:
        raise ValueError(
            "context is too short for symmetric H3 repair plus bidirectional guides"
        )
    repair_start, repair_end, repair_total = min(
        candidates,
        key=lambda item: (
            abs(item[2] - requested_repair),
            item[2] > requested_repair,
            item[2],
        ),
    )
    repair_per_side = repair_total // 2
    left_guide_start = repair_start - guide_frames
    left_guide_end = repair_start
    right_guide_start = repair_end
    right_guide_end = repair_end + guide_frames
    accepted_start = guard
    accepted_end = guard + real_frames
    plan = {
        "schema": SEAM_SCHEMA,
        "fps": int(H3_FPS),
        "left_total_frames": left_count,
        "right_total_frames": right_count,
        "context_frames_per_side": context,
        "repair_requested_frames_total": requested_repair,
        "repair_frames_per_side": repair_per_side,
        "repair_total_frames": repair_total,
        "guide_frames": guide_frames,
        "left_guide_start_frame": left_guide_start,
        "left_guide_end_frame": left_guide_end,
        "right_guide_start_frame": right_guide_start,
        "right_guide_end_frame": right_guide_end,
        "working_frames": working_frames,
        "guard_frames_per_side": guard,
        "cut_frame": cut,
        "repair_start_frame": repair_start,
        "repair_end_frame": repair_end,
        "sampling_start_frame": repair_start,
        "sampling_end_frame": repair_end,
        "sampling_total_frames": repair_total,
        "accepted_start_frame": accepted_start,
        "accepted_end_frame": accepted_end,
        "accepted_frames": real_frames,
        "working_duration_seconds": float(frames_to_seconds(working_frames)),
        "accepted_duration_seconds": float(frames_to_seconds(real_frames)),
        "repair_requested_duration_seconds": float(
            frames_to_seconds(requested_repair)
        ),
        "repair_duration_seconds": float(frames_to_seconds(repair_total)),
        "sampling_duration_seconds": float(frames_to_seconds(repair_total)),
        "left_source_start_frame": left_count - context,
        "right_source_end_frame": context,
    }
    plan["hash"] = content_hash(plan)
    return plan


def make_native_latent_seam_plan(
    left_frame_count: int,
    right_frame_count: int,
    *,
    context_frames_per_side: int = 39,
    working_frames: int = 124,
    accepted_repair_frames: int | None = None,
) -> dict[str, Any]:
    """Plan a phase-safe H3 seam using clean source AV latents directly.

    The target is divided into protected source context, a generated center,
    and protected destination context. Both context clips begin at visual
    token-cycle phase zero, so no latent row is reinterpreted with a different
    causal VAE span. With the production defaults this is exactly
    ``39 protected + 46 generated + 39 protected = 124`` visible frames.
    ``accepted_repair_frames`` may retain only a symmetric inner subset of the
    sampled interval. This gives the sampler temporal overscan while keeping the
    duration-preserving splice independently bounded.
    """

    left_count = int(left_frame_count)
    right_count = int(right_frame_count)
    context = int(context_frames_per_side)
    target_frames = int(working_frames)
    if context not in NATIVE_SEAM_CONTEXT_FRAMES:
        raise ValueError(
            "context_frames_per_side must be one of "
            f"{NATIVE_SEAM_CONTEXT_FRAMES}"
        )
    if target_frames not in NATIVE_SEAM_WORKING_FRAMES:
        raise ValueError(
            "working_frames must be an even trained H3 length: "
            f"{NATIVE_SEAM_WORKING_FRAMES}"
        )
    if not is_h3_frame_count(left_count) or not is_h3_frame_count(right_count):
        raise ValueError("native seam sources must be complete H3 runs on the 17k+5 grid")

    half = target_frames // 2
    if left_count < half or right_count < half:
        raise ValueError(
            f"each source needs at least {half} visible frames for this native seam"
        )
    context_tokens = visual_token_count_for_span(context)
    target_tokens = h3_visual_latent_frames(target_frames)
    right_context_start_token = target_tokens - context_tokens
    if right_context_start_token <= context_tokens:
        raise ValueError("native seam working domain leaves no generated center")
    if right_context_start_token % 5 != 0:
        raise ValueError(
            "destination context would begin at a different H3 visual-token phase"
        )

    left_source_tokens = h3_visual_latent_frames(left_count)
    left_source_start_token = left_source_tokens - context_tokens
    if left_source_start_token % 5 != 0:
        raise ValueError(
            "source tail does not begin at H3 visual-token-cycle phase zero"
        )
    generated_start = context
    generated_end = visual_span_for_tokens(right_context_start_token)
    if generated_end != target_frames - context:
        raise RuntimeError("native seam token and visible-frame geometry disagree")
    generated_total = generated_end - generated_start
    if generated_total < 2 or generated_total % 2:
        raise ValueError("native seam generated interval must split evenly across the cut")
    if accepted_repair_frames is None:
        repair_total = generated_total
    else:
        repair_total = int(accepted_repair_frames)
        if repair_total < 2 or repair_total > generated_total:
            raise ValueError(
                "accepted_repair_frames must be between 2 and the generated interval "
                f"({generated_total})"
            )
        if repair_total % 2:
            raise ValueError("accepted_repair_frames must split evenly across the cut")
    overscan_total = generated_total - repair_total
    if overscan_total % 2:
        raise ValueError("native seam overscan must be symmetric")
    overscan_per_side = overscan_total // 2
    repair_start = generated_start + overscan_per_side
    repair_end = generated_end - overscan_per_side
    repair_per_side = repair_total // 2
    left_guide_start = 0
    left_guide_end = context
    right_guide_start = target_frames - context
    right_guide_end = target_frames

    plan = {
        "schema": SEAM_SCHEMA,
        "mode": "native_av_latent_bidirectional",
        "fps": int(H3_FPS),
        "left_total_frames": left_count,
        "right_total_frames": right_count,
        "context_frames_per_side": context,
        "context_tokens_per_side": context_tokens,
        "working_frames": target_frames,
        "working_video_tokens": target_tokens,
        "guard_frames_per_side": 0,
        "cut_frame": half,
        "repair_requested_frames_total": repair_total,
        "repair_frames_per_side": repair_per_side,
        "repair_total_frames": repair_total,
        "repair_start_frame": repair_start,
        "repair_end_frame": repair_end,
        "sampling_start_frame": generated_start,
        "sampling_end_frame": generated_end,
        "sampling_total_frames": generated_total,
        "overscan_frames_per_side": overscan_per_side,
        "guide_frames": context,
        "left_guide_start_frame": left_guide_start,
        "left_guide_end_frame": left_guide_end,
        "right_guide_start_frame": right_guide_start,
        "right_guide_end_frame": right_guide_end,
        "accepted_start_frame": 0,
        "accepted_end_frame": target_frames,
        "accepted_frames": target_frames,
        "working_duration_seconds": float(frames_to_seconds(target_frames)),
        "accepted_duration_seconds": float(frames_to_seconds(target_frames)),
        "repair_requested_duration_seconds": float(frames_to_seconds(repair_total)),
        "repair_duration_seconds": float(frames_to_seconds(repair_total)),
        "sampling_duration_seconds": float(frames_to_seconds(generated_total)),
        "left_working_source_start_frame": left_count - half,
        "right_working_source_end_frame": half,
        "left_latent_source_start_frame": left_count - context,
        "right_latent_source_end_frame": context,
        "left_latent_source_start_token": left_source_start_token,
        "left_target_start_token": 0,
        "left_target_end_token": context_tokens,
        "right_source_start_token": 0,
        "right_source_end_token": context_tokens,
        "right_target_start_token": right_context_start_token,
        "right_target_end_token": target_tokens,
    }
    plan["hash"] = content_hash(plan)
    return plan


def build_native_latent_seam_window(
    left_frames: Any,
    right_frames: Any,
    plan: dict[str, Any],
):
    """Build the decoded seam domain without guard duplication."""

    _validate_native_latent_seam(plan)
    if int(left_frames.shape[0]) != int(plan["left_total_frames"]):
        raise ValueError("left image batch no longer matches the native seam plan")
    if int(right_frames.shape[0]) != int(plan["right_total_frames"]):
        raise ValueError("right image batch no longer matches the native seam plan")
    if tuple(left_frames.shape[1:]) != tuple(right_frames.shape[1:]):
        raise ValueError("left and right videos must share resolution and channel layout")
    half = int(plan["cut_frame"])
    working = _concat_image_batches([left_frames[-half:], right_frames[:half]])
    if int(working.shape[0]) != int(plan["working_frames"]):
        raise RuntimeError("native seam working image batch has an unexpected length")
    return working


def make_seam_window(plan: dict[str, Any]) -> dict[str, Any]:
    """Compile the working seam domain into the matching exact H3 window."""

    _validate_seam(plan)
    working_frames = int(plan["working_frames"])
    window = make_window(
        f"temporal-inpaint-{str(plan['hash'])[:12]}",
        0,
        frames_to_seconds(working_frames),
        context_frames=0,
        duplicate_prefix_frames=0,
        snap_mode="nearest",
        accept_mode="full_render",
        maximum_frames=working_frames,
    )
    if int(window["shape"]["pixel_frames"]) != working_frames:
        raise RuntimeError("seam plan and generated H3 window disagree")
    return window


def _validate_seam(plan: dict[str, Any]) -> None:
    if plan.get("schema") != SEAM_SCHEMA:
        raise ValueError(f"seam schema must be {SEAM_SCHEMA}")
    if not is_h3_frame_count(int(plan["working_frames"])):
        raise ValueError("seam working frame count is not a legal H3 run")


def _validate_native_latent_seam(plan: dict[str, Any]) -> None:
    _validate_seam(plan)
    if plan.get("mode") != "native_av_latent_bidirectional":
        raise ValueError("seam plan is not a native AV-latent seam")
    context_tokens = int(plan["context_tokens_per_side"])
    target_tokens = int(plan["working_video_tokens"])
    if int(plan["left_target_end_token"]) != context_tokens:
        raise ValueError("native seam left context token range is inconsistent")
    if int(plan["right_target_start_token"]) != target_tokens - context_tokens:
        raise ValueError("native seam right context token range is inconsistent")


def build_seam_window(left_frames: Any, right_frames: Any, plan: dict[str, Any]):
    """Take an opaque tail/head pair and add symmetric H3 guard frames."""

    _validate_seam(plan)
    if int(left_frames.shape[0]) != int(plan["left_total_frames"]):
        raise ValueError("left image batch no longer matches the seam plan")
    if int(right_frames.shape[0]) != int(plan["right_total_frames"]):
        raise ValueError("right image batch no longer matches the seam plan")
    if tuple(left_frames.shape[1:]) != tuple(right_frames.shape[1:]):
        raise ValueError("left and right videos must share resolution and channel layout")
    context = int(plan["context_frames_per_side"])
    guard = int(plan["guard_frames_per_side"])
    left_tail = left_frames[-context:]
    right_head = right_frames[:context]
    parts: list[Any] = []
    if guard:
        parts.extend([left_tail[:1]] * guard)
    parts.extend([left_tail, right_head])
    if guard:
        parts.extend([right_head[-1:]] * guard)
    working = _concat_image_batches(parts)
    if int(working.shape[0]) != int(plan["working_frames"]):
        raise RuntimeError("constructed seam window has an unexpected frame count")
    return working


def seam_video_token_values(
    plan: dict[str, Any],
    projection: str = "cover",
    generation_support: Any = None,
    threshold: float = 0.5,
) -> tuple[float, ...]:
    """Project visible repair support onto exact H3 temporal tokens."""

    _validate_seam(plan)
    if projection not in TOKEN_PROJECTIONS:
        raise ValueError(f"token projection must be one of {', '.join(TOKEN_PROJECTIONS)}")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("sampling threshold must lie in [0, 1]")
    frames = int(plan["working_frames"])
    if generation_support is None:
        strengths = [0.0] * frames
        start = int(plan["sampling_start_frame"])
        end = int(plan["sampling_end_frame"])
        strengths[start:end] = [1.0] * (end - start)
    else:
        if len(generation_support) != frames:
            raise ValueError("generation support must match the working frame count")
        strengths = [min(1.0, max(0.0, float(value))) for value in generation_support]
    values = []
    for token_start, token_end in visual_token_spans(
        h3_visual_latent_frames(int(plan["working_frames"]))
    ):
        support = strengths[token_start:token_end]
        coverage = sum(value >= float(threshold) for value in support)
        if projection == "cover":
            values.append(1.0 if coverage > 0 else 0.0)
        else:
            values.append(1.0 if support and coverage * 2 >= len(support) else 0.0)
    if visual_token_spans(len(values))[-1][1] != int(plan["working_frames"]):
        raise RuntimeError("seam mask and H3 token geometry disagree")
    return tuple(values)


def seam_visible_frame_values(
    plan: dict[str, Any],
    *,
    decoded_blend_frames: int = 8,
    curve: str = "cosine",
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    """Return binary sampling/acceptance and soft decoded opacity fields."""

    _validate_seam(plan)
    frames = int(plan["working_frames"])
    start = int(plan["repair_start_frame"])
    end = int(plan["repair_end_frame"])
    sampling_start = int(plan["sampling_start_frame"])
    sampling_end = int(plan["sampling_end_frame"])

    sampling_values = [0.0] * frames
    sampling_values[sampling_start:sampling_end] = [1.0] * (
        sampling_end - sampling_start
    )
    acceptance_values = [0.0] * frames
    acceptance_values[start:end] = [1.0] * (end - start)
    blend_values = [0.0] * frames
    blend_values[start:end] = _symmetric_weights(end - start, decoded_blend_frames, curve)
    return tuple(sampling_values), tuple(acceptance_values), tuple(blend_values)


def temporal_inpaint_fields(
    working_images: Any,
    plan: dict[str, Any],
    *,
    decoded_blend_frames: int = 8,
    curve: str = "cosine",
):
    """Build visible sampling, acceptance, and output-opacity fields."""

    _validate_seam(plan)
    frames = int(plan["working_frames"])
    if int(working_images.shape[0]) != frames:
        raise ValueError("working image batch does not match the seam plan")
    height, width = int(working_images.shape[1]), int(working_images.shape[2])
    start = int(plan["repair_start_frame"])
    end = int(plan["repair_end_frame"])
    sampling_values, acceptance_values, blend_values = seam_visible_frame_values(
        plan,
        decoded_blend_frames=decoded_blend_frames,
        curve=curve,
    )

    shape = (frames, 1, 1)
    try:
        import torch

        if isinstance(working_images, torch.Tensor):
            device = working_images.device
            sampling = torch.tensor(sampling_values, dtype=torch.float32, device=device)
            acceptance = torch.tensor(
                acceptance_values, dtype=torch.float32, device=device
            )
            blend = torch.tensor(blend_values, dtype=torch.float32, device=device)
            sampling = sampling.view(shape).expand(frames, height, width).clone()
            acceptance = acceptance.view(shape).expand(frames, height, width).clone()
            blend = blend.view(shape).expand(frames, height, width).clone()
            sampling_mean = float(sampling.mean().item())
        else:
            raise TypeError
    except (ImportError, TypeError):
        import numpy

        sampling = numpy.broadcast_to(
            numpy.asarray(sampling_values, dtype=numpy.float32).reshape(shape),
            (frames, height, width),
        ).copy()
        acceptance = numpy.broadcast_to(
            numpy.asarray(acceptance_values, dtype=numpy.float32).reshape(shape),
            (frames, height, width),
        ).copy()
        blend = numpy.broadcast_to(
            numpy.asarray(blend_values, dtype=numpy.float32).reshape(shape),
            (frames, height, width),
        ).copy()
        sampling_mean = float(sampling.mean())
    report = {
        "schema": "cauce.temporal-inpaint-fields/1",
        "seam_hash": plan["hash"],
        "curve": curve,
        "decoded_blend_frames": int(decoded_blend_frames),
        "sampling_mean": sampling_mean,
        "sampling_mode": "h3_per_token_temporal_mask",
        "sampling_range": [
            int(plan["sampling_start_frame"]),
            int(plan["sampling_end_frame"]),
        ],
        "acceptance_frames": end - start,
        "hard_acceptance_range": [start, end],
    }
    return sampling, acceptance, blend, report


def _project_visible_strength(
    strength: Any,
    plan: dict[str, Any],
    video: Any,
    projection: str,
    threshold: float,
):
    """Project an arbitrary visible MASK to explicit binary H3 support."""

    import torch

    if projection not in TOKEN_PROJECTIONS:
        raise ValueError(f"token projection must be one of {', '.join(TOKEN_PROJECTIONS)}")
    frames = int(plan["working_frames"])
    mask = _resize_visible_mask(
        strength,
        frames,
        int(video.shape[3]),
        int(video.shape[4]),
    ).to(device=video.device, dtype=torch.float32)
    hard = torch.zeros((frames, 1, 1), dtype=torch.float32, device=video.device)
    hard[int(plan["sampling_start_frame"]) : int(plan["sampling_end_frame"])] = 1.0
    mask = (mask * hard >= float(threshold)).to(dtype=torch.float32)
    reduced = []
    for token_start, token_end in visual_token_spans(int(video.shape[2])):
        support = mask[token_start:token_end]
        if projection == "cover":
            reduced.append(support.amax(dim=0))
        else:
            reduced.append((support.mean(dim=0) >= 0.5).to(dtype=torch.float32))
    return torch.stack(reduced, dim=0)


def seam_splice_ranges(plan: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """Return the exact source/patch ranges used by the duration-preserving splice."""

    _validate_seam(plan)
    repair = int(plan["repair_frames_per_side"])
    left_count = int(plan["left_total_frames"])
    right_count = int(plan["right_total_frames"])
    return {
        "left_keep": (0, left_count - repair),
        "working_patch": (
            int(plan["repair_start_frame"]),
            int(plan["repair_end_frame"]),
        ),
        "right_keep": (repair, right_count),
    }


def _video_samples(video_latent: dict[str, Any]):
    samples = video_latent.get("samples")
    if samples is None:
        raise ValueError("encoded video latent has no samples")
    if getattr(samples, "is_nested", False):
        streams = list(samples.unbind())
        if not streams:
            raise ValueError("encoded nested latent is empty")
        samples = streams[0]
    if getattr(samples, "ndim", 0) != 5:
        raise ValueError("encoded H3 video latent must have five dimensions")
    return samples


def prepare_h3_temporal_inpaint(
    target_latent: dict[str, Any],
    encoded_video_latent: dict[str, Any],
    plan: dict[str, Any],
    *,
    projection: str = "cover",
    sampling_threshold: float = 0.5,
    generation_support: Any = None,
):
    """Inject the source video domain and denoise only its central seam."""

    import torch

    try:
        import comfy.nested_tensor  # type: ignore
    except ImportError as exc:  # pragma: no cover - requires ComfyUI
        raise RuntimeError("H3 temporal inpainting can only be prepared inside ComfyUI") from exc

    _validate_seam(plan)
    capabilities = require_h3_temporal_edit_runtime()
    target_video, target_audio = get_av_streams(target_latent)
    encoded_video = _video_samples(encoded_video_latent)
    if pixel_frames_from_video_latent(target_video) != int(plan["working_frames"]):
        raise ValueError("target H3 latent length does not match the seam working domain")
    if tuple(encoded_video.shape) != tuple(target_video.shape):
        raise ValueError(
            "encoded video latent geometry does not match the H3 target; normalize both clips "
            "to the execution profile before building the seam"
        )
    video = encoded_video.to(device=target_video.device, dtype=target_video.dtype).clone()
    audio = target_audio.clone()
    if generation_support is None:
        token_values = seam_video_token_values(
            plan, projection=projection, threshold=sampling_threshold
        )
        video_mask = torch.tensor(
            token_values, dtype=torch.float32, device=video.device
        ).view(1, 1, int(video.shape[2]), 1, 1)
        video_mask = video_mask.expand(
            int(video.shape[0]),
            1,
            int(video.shape[2]),
            int(video.shape[3]),
            int(video.shape[4]),
        ).clone()
        field_source = "plan_token_aligned_repair"
    else:
        projected = _project_visible_strength(
            generation_support, plan, video, projection, sampling_threshold
        )
        video_mask = projected.unsqueeze(0).unsqueeze(0).expand(
            int(video.shape[0]),
            1,
            int(video.shape[2]),
            int(video.shape[3]),
            int(video.shape[4]),
        ).clone()
        token_values = [float(value) for value in projected.mean(dim=(1, 2)).tolist()]
        field_source = "connected_mask"
    audio_mask = torch.zeros(
        (int(audio.shape[0]), 1, int(audio.shape[2]), int(audio.shape[-1])),
        dtype=torch.float32,
        device=audio.device,
    )
    out = copy.copy(target_latent)
    out["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
    out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    report = {
        "schema": "cauce.temporal-inpaint-mask-report/1",
        "seam_hash": plan["hash"],
        "working_frames": int(plan["working_frames"]),
        "repair_start_frame": int(plan["repair_start_frame"]),
        "repair_end_frame": int(plan["repair_end_frame"]),
        "sampling_start_frame": int(plan["sampling_start_frame"]),
        "sampling_end_frame": int(plan["sampling_end_frame"]),
        "token_projection": projection,
        "sampling_threshold": float(sampling_threshold),
        "sampling_mask_mode": "official_h3_per_token_denoise_mask",
        "field_source": field_source,
        "video_generate_tokens": sum(value > 0.0 for value in token_values),
        "video_full_generate_tokens": sum(value >= 1.0 - 1e-6 for value in token_values),
        "video_soft_tokens": 0,
        "video_token_values": list(token_values),
        "audio_mode": "internal-zero-mask-discarded",
        "acceptance_mode": "hard-decoded-splice",
        "runtime_capabilities": capabilities,
    }
    return out, report


def prepare_h3_native_latent_temporal_inpaint(
    target_latent: dict[str, Any],
    left_latent: dict[str, Any],
    right_latent: dict[str, Any],
    plan: dict[str, Any],
):
    """Pin clean source tail/head latents and generate only the central gap."""

    import torch

    try:
        import comfy.nested_tensor  # type: ignore
    except ImportError as exc:  # pragma: no cover - requires ComfyUI
        raise RuntimeError("native H3 temporal inpainting requires ComfyUI") from exc

    _validate_native_latent_seam(plan)
    capabilities = require_h3_temporal_edit_runtime()
    target_video, target_audio = get_av_streams(target_latent)
    left_video, _ = get_av_streams(left_latent)
    right_video, _ = get_av_streams(right_latent)
    if pixel_frames_from_video_latent(target_video) != int(plan["working_frames"]):
        raise ValueError("target H3 latent length does not match the native seam domain")
    if pixel_frames_from_video_latent(left_video) != int(plan["left_total_frames"]):
        raise ValueError("left source latent length does not match the native seam plan")
    if pixel_frames_from_video_latent(right_video) != int(plan["right_total_frames"]):
        raise ValueError("right source latent length does not match the native seam plan")
    expected_geometry = tuple(target_video.shape[:2] + target_video.shape[3:])
    if tuple(left_video.shape[:2] + left_video.shape[3:]) != expected_geometry or tuple(
        right_video.shape[:2] + right_video.shape[3:]
    ) != expected_geometry:
        raise ValueError(
            "native seam latents must share batch, channels, height, and width; "
            "generate both clips with the same CAUCE execution profile"
        )

    left_source_start = int(plan["left_latent_source_start_token"])
    context_tokens = int(plan["context_tokens_per_side"])
    right_target_start = int(plan["right_target_start_token"])
    left_context = left_video[:, :, left_source_start:].to(target_video)
    right_context = right_video[:, :, :context_tokens].to(target_video)
    if int(left_context.shape[2]) != context_tokens or int(right_context.shape[2]) != context_tokens:
        raise RuntimeError("native source-context extraction returned an unexpected token count")

    video = target_video.clone()
    audio = target_audio.clone()
    video[:, :, :context_tokens] = left_context
    video[:, :, right_target_start:] = right_context
    video_mask = torch.ones(
        (
            int(video.shape[0]),
            1,
            int(video.shape[2]),
            int(video.shape[3]),
            int(video.shape[4]),
        ),
        dtype=torch.float32,
        device=video.device,
    )
    video_mask[:, :, :context_tokens] = 0.0
    video_mask[:, :, right_target_start:] = 0.0
    audio_mask = torch.zeros(
        (int(audio.shape[0]), 1, int(audio.shape[2]), int(audio.shape[-1])),
        dtype=torch.float32,
        device=audio.device,
    )
    out = copy.copy(target_latent)
    out["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
    out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    report = {
        "schema": "cauce.native-latent-temporal-inpaint/1",
        "seam_hash": plan["hash"],
        "working_frames": int(plan["working_frames"]),
        "video_tokens": int(video.shape[2]),
        "left_protected_tokens": [0, context_tokens],
        "generated_tokens": [context_tokens, right_target_start],
        "right_protected_tokens": [right_target_start, int(video.shape[2])],
        "left_source_tokens": [left_source_start, int(left_video.shape[2])],
        "right_source_tokens": [0, context_tokens],
        "sampling_mask_mode": "official_h3_per-token_binary_native-latent",
        "video_generate_tokens": right_target_start - context_tokens,
        "video_preserved_tokens": context_tokens * 2,
        "audio_mode": "internal-zero-mask-discarded",
        "decoded_acceptance": "duration-preserving-cosine-splice",
        "runtime_capabilities": capabilities,
    }
    return out, report


def add_h3_temporal_inpaint_guides(
    positive: Any,
    target_latent: dict[str, Any],
    working_images: Any,
    plan: dict[str, Any],
    vae: Any,
):
    """Anchor preserved motion clips immediately before and after the gap."""

    _validate_seam(plan)
    capabilities = require_h3_temporal_edit_runtime()
    if int(working_images.shape[0]) != int(plan["working_frames"]):
        raise ValueError("working image batch does not match the seam plan")
    left_start = int(plan["left_guide_start_frame"])
    left_end = int(plan["left_guide_end_frame"])
    right_start = int(plan["right_guide_start_frame"])
    right_end = int(plan["right_guide_end_frame"])
    left_clip = working_images[left_start:left_end]
    right_clip = working_images[right_start:right_end]
    if int(left_clip.shape[0]) != int(plan["guide_frames"]) or int(
        right_clip.shape[0]
    ) != int(plan["guide_frames"]):
        raise RuntimeError("temporal inpaint guide extraction produced an invalid H3 clip")
    conditioned = execute_add_guide(
        positive=positive,
        latent=target_latent,
        frame_idx=left_start,
        vae=vae,
        image=left_clip,
    )
    conditioned = execute_add_guide(
        positive=conditioned,
        latent=target_latent,
        frame_idx=right_start,
        vae=vae,
        image=right_clip,
    )
    report = {
        "schema": "cauce.temporal-inpaint-guides/1",
        "seam_hash": plan["hash"],
        "guide_frames": int(plan["guide_frames"]),
        "left_range": [left_start, left_end],
        "right_range": [right_start, right_end],
        "generated_range": [
            int(plan["repair_start_frame"]),
            int(plan["repair_end_frame"]),
        ],
        "audio_mode": "no-guide-audio",
        "runtime_capabilities": capabilities,
    }
    return conditioned, report


def splice_seam_patch(
    left_frames: Any,
    right_frames: Any,
    repaired_working_frames: Any,
    plan: dict[str, Any],
    *,
    feather_frames: int = 8,
    curve: str = "cosine",
    blend_strength: Any = None,
):
    """Replace only the inner tail/head frames and preserve total duration."""

    _validate_seam(plan)
    if int(left_frames.shape[0]) != int(plan["left_total_frames"]):
        raise ValueError("left image batch no longer matches the seam plan")
    if int(right_frames.shape[0]) != int(plan["right_total_frames"]):
        raise ValueError("right image batch no longer matches the seam plan")
    if int(repaired_working_frames.shape[0]) != int(plan["working_frames"]):
        raise ValueError("repaired image batch does not match the seam working domain")
    if tuple(left_frames.shape[1:]) != tuple(right_frames.shape[1:]) or tuple(
        left_frames.shape[1:]
    ) != tuple(repaired_working_frames.shape[1:]):
        raise ValueError("source and repaired videos must share resolution and channels")
    repair = int(plan["repair_frames_per_side"])
    ranges = seam_splice_ranges(plan)
    start, end = ranges["working_patch"]
    patch = repaired_working_frames[start:end]
    original = _concat_image_batches([left_frames[-repair:], right_frames[:repair]])
    if blend_strength is not None:
        if int(blend_strength.shape[0]) == int(plan["working_frames"]):
            blend_strength = blend_strength[start:end]
        elif int(blend_strength.shape[0]) not in {1, int(patch.shape[0])}:
            raise ValueError(
                "blend mask must have one frame, the patch length, or the working length"
            )
    patch = _blend_patch(
        patch,
        original,
        feather_frames,
        curve,
        blend_strength=blend_strength,
    )
    left_start, left_end = ranges["left_keep"]
    right_start, right_end = ranges["right_keep"]
    joined = _concat_image_batches(
        [left_frames[left_start:left_end], patch, right_frames[right_start:right_end]]
    )
    expected = int(left_frames.shape[0]) + int(right_frames.shape[0])
    if int(joined.shape[0]) != expected:
        raise RuntimeError("duration-preserving seam splice changed the frame count")
    report = {
        "schema": "cauce.seam-splice-report/1",
        "seam_hash": plan["hash"],
        "left_frames": int(left_frames.shape[0]),
        "right_frames": int(right_frames.shape[0]),
        "output_frames": int(joined.shape[0]),
        "replacement_frames": int(patch.shape[0]),
        "decoded_feather_frames": int(feather_frames),
        "blend_curve": curve,
        "blend_source": "connected_mask" if blend_strength is not None else "temporal_curve",
    }
    return joined, patch, report


def assemble_native_two_clip_loop(
    first_clip: Any,
    second_clip: Any,
    forward_repaired_working: Any,
    forward_plan: dict[str, Any],
    wrap_repaired_working: Any,
    wrap_plan: dict[str, Any],
    *,
    feather_frames: int = 4,
    curve: str = "cosine",
):
    """Apply first→second and second→first seam proposals as a closed loop."""

    _validate_native_latent_seam(forward_plan)
    _validate_native_latent_seam(wrap_plan)
    first_count = int(first_clip.shape[0])
    second_count = int(second_clip.shape[0])
    if (first_count, second_count) != (
        int(forward_plan["left_total_frames"]),
        int(forward_plan["right_total_frames"]),
    ):
        raise ValueError("forward seam plan does not match first→second clips")
    if (second_count, first_count) != (
        int(wrap_plan["left_total_frames"]),
        int(wrap_plan["right_total_frames"]),
    ):
        raise ValueError("wrap seam plan does not match second→first clips")
    shapes = {
        tuple(first_clip.shape[1:]),
        tuple(second_clip.shape[1:]),
        tuple(forward_repaired_working.shape[1:]),
        tuple(wrap_repaired_working.shape[1:]),
    }
    if len(shapes) != 1:
        raise ValueError("all source and repaired videos must share resolution and channels")
    for repaired, plan, label in (
        (forward_repaired_working, forward_plan, "forward"),
        (wrap_repaired_working, wrap_plan, "wrap"),
    ):
        if int(repaired.shape[0]) != int(plan["working_frames"]):
            raise ValueError(f"{label} repaired batch does not match its seam plan")

    forward_repair = int(forward_plan["repair_frames_per_side"])
    wrap_repair = int(wrap_plan["repair_frames_per_side"])
    if forward_repair != wrap_repair:
        raise ValueError("closed-loop seams must replace the same frames per side")
    repair = forward_repair
    if repair * 2 >= min(first_count, second_count):
        raise ValueError("seam patches overlap; use longer source clips or a smaller repair")

    forward_start = int(forward_plan["repair_start_frame"])
    forward_end = int(forward_plan["repair_end_frame"])
    wrap_start = int(wrap_plan["repair_start_frame"])
    wrap_end = int(wrap_plan["repair_end_frame"])
    forward_patch = forward_repaired_working[forward_start:forward_end]
    wrap_patch = wrap_repaired_working[wrap_start:wrap_end]
    forward_original = _concat_image_batches(
        [first_clip[-repair:], second_clip[:repair]]
    )
    wrap_original = _concat_image_batches([second_clip[-repair:], first_clip[:repair]])
    forward_patch = _blend_patch(
        forward_patch, forward_original, feather_frames, curve
    )
    wrap_patch = _blend_patch(wrap_patch, wrap_original, feather_frames, curve)

    first_repaired = _concat_image_batches(
        [
            wrap_patch[repair:],
            first_clip[repair : first_count - repair],
            forward_patch[:repair],
        ]
    )
    second_repaired = _concat_image_batches(
        [
            forward_patch[repair:],
            second_clip[repair : second_count - repair],
            wrap_patch[:repair],
        ]
    )
    loop = _concat_image_batches([first_repaired, second_repaired])
    if int(first_repaired.shape[0]) != first_count or int(
        second_repaired.shape[0]
    ) != second_count:
        raise RuntimeError("native seam assembly changed an individual clip duration")
    if int(loop.shape[0]) != first_count + second_count:
        raise RuntimeError("native seam assembly changed the loop duration")
    report = {
        "schema": "cauce.native-two-clip-loop/1",
        "first_frames": first_count,
        "second_frames": second_count,
        "loop_frames": int(loop.shape[0]),
        "forward_seam_hash": forward_plan["hash"],
        "wrap_seam_hash": wrap_plan["hash"],
        "replacement_frames_per_side": repair,
        "decoded_feather_frames": int(feather_frames),
        "blend_curve": curve,
        "loop_boundary": "repaired-second-to-first",
    }
    return loop, first_repaired, second_repaired, forward_patch, wrap_patch, report
