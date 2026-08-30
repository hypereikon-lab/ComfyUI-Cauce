"""Low-level, timeline-aware operations for packed MiniMax H3 AV latents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from typing import Any, Callable

from .contracts import AV_SPAN_SCHEMA, AV_WINDOW_LAYOUT_SCHEMA, content_hash
from .h3 import get_av_streams, validate_av_latent
from .timebase import (
    ceil_h3_frame_count,
    h3_audio_token_boundary,
    h3_visual_latent_frames,
    is_h3_frame_count,
    visual_token_boundary,
    visual_token_spans,
)


NestedFactory = Callable[[tuple[Any, Any]], Any]


def _clone(value: Any):
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    copy = getattr(value, "copy", None)
    if callable(copy):
        return copy()
    raise TypeError("AV tensors must provide clone() or copy()")


def _new_zeros(reference: Any, shape: tuple[int, ...]):
    new_zeros = getattr(reference, "new_zeros", None)
    if callable(new_zeros):
        return new_zeros(shape)
    try:
        import numpy as np

        return np.zeros(shape, dtype=reference.dtype)
    except (ImportError, AttributeError) as exc:  # pragma: no cover - NumPy ships with ComfyUI
        raise TypeError("AV tensors must support new_zeros() or be NumPy arrays") from exc


def _new_full(reference: Any, shape: tuple[int, ...], value: float):
    new_full = getattr(reference, "new_full", None)
    if callable(new_full):
        return new_full(shape, float(value))
    try:
        import numpy as np

        return np.full(shape, float(value), dtype=reference.dtype)
    except (ImportError, AttributeError) as exc:  # pragma: no cover - NumPy ships with ComfyUI
        raise TypeError("AV tensors must support new_full() or be NumPy arrays") from exc


def _concatenate(values: tuple[Any, ...], axis: int):
    first = values[0]
    try:
        import torch

        if isinstance(first, torch.Tensor):
            return torch.cat(values, dim=axis)
    except ImportError:  # pragma: no cover - PyTorch ships with ComfyUI
        pass
    try:
        import numpy as np

        if isinstance(first, np.ndarray):
            return np.concatenate(values, axis=axis)
    except ImportError:  # pragma: no cover - NumPy ships with ComfyUI
        pass
    raise TypeError("AV tensors must be PyTorch tensors or NumPy arrays")


def _resize_spatial(value: Any, height: int, width: int, method: str):
    """Resize only the last two dimensions of a tensor, never its time axis."""

    target = (int(height), int(width))
    if min(target) < 1:
        raise ValueError("spatial target dimensions must be positive")
    if tuple(int(item) for item in value.shape[-2:]) == target:
        return _clone(value)
    modes = {"nearest-exact", "bilinear", "bicubic", "area"}
    if method not in modes:
        raise ValueError(f"resize method must be one of {sorted(modes)}")
    try:
        import torch
        import torch.nn.functional as functional

        if isinstance(value, torch.Tensor):
            source_shape = tuple(int(item) for item in value.shape)
            flat = value.reshape((-1, 1, source_shape[-2], source_shape[-1]))
            kwargs = {}
            if method in {"bilinear", "bicubic"}:
                kwargs["align_corners"] = False
            resized = functional.interpolate(flat, size=target, mode=method, **kwargs)
            return resized.reshape(source_shape[:-2] + target)
    except ImportError:  # pragma: no cover - PyTorch ships with ComfyUI
        pass
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            # NumPy is only the lightweight test backend. Runtime interpolation
            # uses PyTorch; nearest-neighbour keeps shape and placement tests exact.
            ys = np.rint(np.linspace(0, value.shape[-2] - 1, target[0])).astype(np.int64)
            xs = np.rint(np.linspace(0, value.shape[-1] - 1, target[1])).astype(np.int64)
            return np.take(np.take(value, ys, axis=-2), xs, axis=-1)
    except ImportError:  # pragma: no cover - NumPy ships with ComfyUI
        pass
    raise TypeError("spatial resizing requires PyTorch or NumPy tensors")


def _make_latent(video: Any, audio: Any, nested_factory: NestedFactory | None):
    samples = nested_factory((video, audio)) if nested_factory is not None else (video, audio)
    return {"samples": samples}


def _with_streams(
    latent: Mapping[str, Any],
    video: Any,
    audio: Any,
    nested_factory: NestedFactory | None,
) -> dict[str, Any]:
    out = dict(latent)
    out["samples"] = nested_factory((video, audio)) if nested_factory is not None else (video, audio)
    return out


def _profile_tensor(reference: Any, values: Sequence[float]):
    try:
        import torch

        if isinstance(reference, torch.Tensor):
            return torch.tensor(values, dtype=torch.float32, device=reference.device)
    except ImportError:  # pragma: no cover - PyTorch ships with ComfyUI
        pass
    try:
        import numpy as np

        if isinstance(reference, np.ndarray):
            return np.asarray(values, dtype=np.float32)
    except ImportError:  # pragma: no cover - NumPy ships with ComfyUI
        pass
    raise TypeError("AV tensors must be PyTorch tensors or NumPy arrays")


def _mask_tensor(reference: Any, value: Any):
    """Return one float32 mask on the reference backend and device."""

    try:
        import torch

        if isinstance(reference, torch.Tensor):
            if isinstance(value, torch.Tensor):
                return value.to(device=reference.device, dtype=torch.float32)
            return torch.as_tensor(value, device=reference.device, dtype=torch.float32)
    except ImportError:  # pragma: no cover - PyTorch ships with ComfyUI
        pass
    try:
        import numpy as np

        if isinstance(reference, np.ndarray):
            return np.asarray(value, dtype=np.float32)
    except ImportError:  # pragma: no cover - NumPy ships with ComfyUI
        pass
    raise TypeError("mask and AV tensors must be PyTorch tensors or NumPy arrays")


def _mask_min_max(value: Any) -> tuple[float, float]:
    try:
        return float(value.min().item()), float(value.max().item())
    except AttributeError:
        return float(value.min()), float(value.max())


def _mask_all_finite(value: Any) -> bool:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all().item())
    except ImportError:  # pragma: no cover - PyTorch ships with ComfyUI
        pass
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return bool(np.isfinite(value).all())
    except ImportError:  # pragma: no cover - NumPy ships with ComfyUI
        pass
    raise TypeError("mask tensors must be PyTorch tensors or NumPy arrays")


def _mask_digest(value: Any) -> str:
    """Hash one mask after canonical float32 CPU materialization."""

    try:
        import torch

        if isinstance(value, torch.Tensor):
            array = value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        else:
            array = value
    except ImportError:  # pragma: no cover - PyTorch ships with ComfyUI
        array = value
    try:
        import numpy as np

        canonical = np.asarray(array, dtype=np.float32)
        digest = hashlib.sha256()
        digest.update(str(tuple(int(item) for item in canonical.shape)).encode("ascii"))
        digest.update(canonical.tobytes(order="C"))
        return digest.hexdigest()
    except ImportError as exc:  # pragma: no cover - NumPy ships with ComfyUI
        raise TypeError("mask hashing requires NumPy") from exc


def _resize_mask_frames(mask: Any, height: int, width: int):
    """Resize ``[N,H,W]`` masks while preserving continuous values."""

    target = (int(height), int(width))
    if tuple(int(item) for item in mask.shape[-2:]) == target:
        return mask.clone() if hasattr(mask, "clone") else mask.copy()
    try:
        import torch
        import torch.nn.functional as functional

        if isinstance(mask, torch.Tensor):
            return functional.interpolate(
                mask.unsqueeze(1),
                size=target,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
    except ImportError:  # pragma: no cover - PyTorch ships with ComfyUI
        pass
    try:
        import numpy as np

        if isinstance(mask, np.ndarray):
            source_h, source_w = (int(mask.shape[-2]), int(mask.shape[-1]))
            ys = np.linspace(0.0, max(0, source_h - 1), target[0], dtype=np.float32)
            xs = np.linspace(0.0, max(0, source_w - 1), target[1], dtype=np.float32)
            y0 = np.floor(ys).astype(np.int64)
            x0 = np.floor(xs).astype(np.int64)
            y1 = np.minimum(y0 + 1, source_h - 1)
            x1 = np.minimum(x0 + 1, source_w - 1)
            wy = (ys - y0).reshape(1, target[0], 1)
            wx = (xs - x0).reshape(1, 1, target[1])
            top = mask[:, y0][:, :, x0] * (1.0 - wx) + mask[:, y0][:, :, x1] * wx
            bottom = mask[:, y1][:, :, x0] * (1.0 - wx) + mask[:, y1][:, :, x1] * wx
            return top * (1.0 - wy) + bottom * wy
    except ImportError:  # pragma: no cover - NumPy ships with ComfyUI
        pass
    raise TypeError("mask resizing requires PyTorch or NumPy tensors")


def _temporal_amax(value: Any, start: int, end: int):
    segment = value[int(start):int(end)]
    maximum = getattr(segment, "amax", None)
    if callable(maximum):
        try:
            return maximum(dim=0)
        except TypeError:
            return maximum(axis=0)
    maximum = getattr(segment, "max", None)
    if callable(maximum):
        try:
            return maximum(dim=0).values
        except TypeError:
            return maximum(axis=0)
    raise TypeError("mask tensors must support amax")


def _broadcast_video_mask(profile: Any, video: Any):
    shape = (int(video.shape[0]), 1, int(video.shape[2]), int(video.shape[3]), int(video.shape[4]))
    try:
        import torch

        if isinstance(profile, torch.Tensor):
            return profile.reshape(1, 1, -1, 1, 1).expand(shape).clone()
    except ImportError:  # pragma: no cover - PyTorch ships with ComfyUI
        pass
    try:
        import numpy as np

        if isinstance(profile, np.ndarray):
            return np.broadcast_to(profile.reshape(1, 1, -1, 1, 1), shape).copy()
    except ImportError:  # pragma: no cover - NumPy ships with ComfyUI
        pass
    raise TypeError("mask profiles must be PyTorch tensors or NumPy arrays")


def _broadcast_audio_mask(profile: Any, audio: Any):
    shape = (int(audio.shape[0]), 1, int(audio.shape[2]), int(audio.shape[3]))
    try:
        import torch

        if isinstance(profile, torch.Tensor):
            return profile.reshape(1, 1, 1, -1).expand(shape).clone()
    except ImportError:  # pragma: no cover - PyTorch ships with ComfyUI
        pass
    try:
        import numpy as np

        if isinstance(profile, np.ndarray):
            return np.broadcast_to(profile.reshape(1, 1, 1, -1), shape).copy()
    except ImportError:  # pragma: no cover - NumPy ships with ComfyUI
        pass
    raise TypeError("mask profiles must be PyTorch tensors or NumPy arrays")


def _combine_mask(existing: Any, proposed: Any, mode: str):
    if mode == "replace" or existing is None:
        return proposed
    if tuple(existing.shape) != tuple(proposed.shape):
        raise ValueError("existing AV noise_mask shape differs from the proposed mask")
    if mode == "maximum":
        maximum = getattr(existing, "maximum", None)
        if callable(maximum):
            return maximum(proposed)
        try:
            import numpy as np

            return np.maximum(existing, proposed)
        except ImportError as exc:  # pragma: no cover - NumPy ships with ComfyUI
            raise TypeError("mask tensors must support maximum") from exc
    if mode == "minimum":
        minimum = getattr(existing, "minimum", None)
        if callable(minimum):
            return minimum(proposed)
        try:
            import numpy as np

            return np.minimum(existing, proposed)
        except ImportError as exc:  # pragma: no cover - NumPy ships with ComfyUI
            raise TypeError("mask tensors must support minimum") from exc
    if mode == "multiply":
        return existing * proposed
    raise ValueError("mask combine mode must be replace, maximum, minimum, or multiply")


def _mask_streams(latent: Mapping[str, Any]) -> tuple[Any | None, Any | None]:
    value = latent.get("noise_mask")
    if value is None:
        return None, None
    try:
        return get_av_streams({"samples": value})
    except (TypeError, ValueError) as exc:
        raise ValueError("H3 noise_mask must contain nested video and audio streams") from exc


def _curve(value: float, curve: str) -> float:
    x = min(1.0, max(0.0, float(value)))
    if curve == "linear":
        return x
    if curve == "smoothstep":
        return x * x * (3.0 - 2.0 * x)
    if curve == "smootherstep":
        return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
    raise ValueError("mask curve must be linear, smoothstep, or smootherstep")


def _interval_weight(
    position: float,
    start: int,
    end: int,
    fade_in: int,
    fade_out: int,
    curve: str,
) -> float:
    if start <= position < end:
        return 1.0
    if fade_in > 0 and start - fade_in < position < start:
        return _curve((position - (start - fade_in)) / fade_in, curve)
    if fade_out > 0 and end <= position < end + fade_out:
        return _curve(1.0 - ((position - end) / fade_out), curve)
    return 0.0


def _dtype(value: Any) -> str:
    return str(getattr(value, "dtype", "unknown"))


def _device(value: Any) -> str:
    return str(getattr(value, "device", "cpu"))


def _validate_tensor_compatibility(
    left_video: Any,
    left_audio: Any,
    right_video: Any,
    right_audio: Any,
) -> None:
    if tuple(left_video.shape[:2]) != tuple(right_video.shape[:2]):
        raise ValueError("H3 video batch/channel dimensions must match")
    if tuple(left_video.shape[3:]) != tuple(right_video.shape[3:]):
        raise ValueError("H3 video spatial dimensions must match")
    if tuple(left_audio.shape[:3]) != tuple(right_audio.shape[:3]):
        raise ValueError("H3 structural-audio batch/channel dimensions must match")
    if getattr(left_video, "dtype", None) != getattr(right_video, "dtype", None):
        raise TypeError("H3 video dtypes must match")
    if getattr(left_audio, "dtype", None) != getattr(right_audio, "dtype", None):
        raise TypeError("H3 structural-audio dtypes must match")
    if _device(left_video) != _device(right_video):
        raise ValueError("H3 video devices must match")
    if _device(left_audio) != _device(right_audio):
        raise ValueError("H3 structural-audio devices must match")


def inspect_av_latent(
    latent: Mapping[str, Any],
    *,
    timeline_origin_frame: int = 0,
) -> dict[str, Any]:
    """Return a serializable report for one complete H3 latent or aligned window."""

    video, audio, frames = validate_av_latent(
        latent,
        timeline_origin_frame=timeline_origin_frame,
    )
    return {
        "schema": "cauce.h3-av-latent-report/1",
        "timeline_origin_frame": int(timeline_origin_frame),
        "timeline_end_frame": int(timeline_origin_frame) + frames,
        "frame_count": frames,
        "video_tokens": int(video.shape[2]),
        "audio_tokens": int(audio.shape[-1]),
        "video_shape": [int(item) for item in video.shape],
        "audio_shape": [int(item) for item in audio.shape],
        "video_dtype": _dtype(video),
        "audio_dtype": _dtype(audio),
        "video_device": _device(video),
        "audio_device": _device(audio),
    }


def plan_av_window(
    previous_av_latent: Mapping[str, Any],
    *,
    overlap_frames: int,
    extension_frames: int,
) -> dict[str, Any]:
    """Plan one globally aligned fresh AV window without assigning workflow intent."""

    _, _, previous_frames = validate_av_latent(previous_av_latent, name="previous_av_latent")
    overlap = int(overlap_frames)
    extension = int(extension_frames)
    if not is_h3_frame_count(overlap):
        raise ValueError("overlap_frames must satisfy the H3 17k+5 grid")
    if overlap > previous_frames:
        raise ValueError("overlap_frames cannot exceed the previous latent")
    if extension < 17 or extension % 17:
        raise ValueError("extension_frames must be a positive multiple of 17")

    window_frames = overlap + extension
    if not is_h3_frame_count(window_frames):
        raise ValueError("the planned window must satisfy the H3 17k+5 grid")
    window_start = previous_frames - overlap
    window_end = previous_frames + extension
    overlap_video_tokens = h3_visual_latent_frames(overlap)
    target_video_tokens = h3_visual_latent_frames(window_frames)
    extension_video_tokens = target_video_tokens - overlap_video_tokens
    overlap_audio_tokens = h3_audio_token_boundary(previous_frames) - h3_audio_token_boundary(
        window_start
    )
    extension_audio_tokens = h3_audio_token_boundary(window_end) - h3_audio_token_boundary(
        previous_frames
    )
    target_audio_tokens = overlap_audio_tokens + extension_audio_tokens
    payload: dict[str, Any] = {
        "schema": AV_WINDOW_LAYOUT_SCHEMA,
        "previous_frame_count": previous_frames,
        "window_start_frame": window_start,
        "window_end_frame": window_end,
        "window_frame_count": window_frames,
        "overlap_frames": overlap,
        "extension_frames": extension,
        "target_video_tokens": target_video_tokens,
        "target_audio_tokens": target_audio_tokens,
        "overlap_video_tokens": overlap_video_tokens,
        "overlap_audio_tokens": overlap_audio_tokens,
        "extension_video_tokens": extension_video_tokens,
        "extension_audio_tokens": extension_audio_tokens,
    }
    payload["layout_hash"] = content_hash(payload)
    return payload


def validate_av_window_layout(layout: Mapping[str, Any]) -> None:
    if not isinstance(layout, Mapping) or layout.get("schema") != AV_WINDOW_LAYOUT_SCHEMA:
        raise ValueError(f"AV window layout must use schema {AV_WINDOW_LAYOUT_SCHEMA}")
    supplied_hash = layout.get("layout_hash")
    unhashed = {key: value for key, value in layout.items() if key != "layout_hash"}
    if supplied_hash != content_hash(unhashed):
        raise ValueError("AV window layout hash does not match its contents")

    fields = (
        "previous_frame_count",
        "window_start_frame",
        "window_end_frame",
        "window_frame_count",
        "overlap_frames",
        "extension_frames",
        "target_video_tokens",
        "target_audio_tokens",
        "overlap_video_tokens",
        "overlap_audio_tokens",
        "extension_video_tokens",
        "extension_audio_tokens",
    )
    try:
        values = {field: int(layout[field]) for field in fields}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("AV window layout is missing an integer field") from exc
    previous = values["previous_frame_count"]
    overlap = values["overlap_frames"]
    extension = values["extension_frames"]
    if not is_h3_frame_count(previous):
        raise ValueError("AV window previous_frame_count must satisfy the H3 17k+5 grid")
    if not is_h3_frame_count(overlap) or overlap > previous:
        raise ValueError("AV window overlap must be a valid H3 run within the previous latent")
    if extension < 17 or extension % 17:
        raise ValueError("AV window extension must be a positive multiple of 17")
    window = overlap + extension
    start = previous - overlap
    end = previous + extension
    expected = {
        "window_start_frame": start,
        "window_end_frame": end,
        "window_frame_count": window,
        "target_video_tokens": h3_visual_latent_frames(window),
        "target_audio_tokens": h3_audio_token_boundary(end)
        - h3_audio_token_boundary(start),
        "overlap_video_tokens": h3_visual_latent_frames(overlap),
        "overlap_audio_tokens": h3_audio_token_boundary(previous)
        - h3_audio_token_boundary(start),
    }
    expected["extension_video_tokens"] = (
        expected["target_video_tokens"] - expected["overlap_video_tokens"]
    )
    expected["extension_audio_tokens"] = h3_audio_token_boundary(
        end
    ) - h3_audio_token_boundary(previous)
    for field, expected_value in expected.items():
        if values[field] != expected_value:
            raise ValueError(
                f"AV window layout {field} must be {expected_value}, got {values[field]}"
            )


def allocate_av_window_like(
    previous_av_latent: Mapping[str, Any],
    layout: Mapping[str, Any],
    *,
    nested_factory: NestedFactory | None = None,
) -> dict[str, Any]:
    """Allocate the zero AV target described by a validated absolute layout."""

    validate_av_window_layout(layout)
    video, audio, previous_frames = validate_av_latent(
        previous_av_latent,
        name="previous_av_latent",
    )
    if previous_frames != int(layout["previous_frame_count"]):
        raise ValueError("previous latent length differs from the AV window layout")
    target_video = _new_zeros(
        video,
        (
            int(video.shape[0]),
            int(video.shape[1]),
            int(layout["target_video_tokens"]),
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
            int(layout["target_audio_tokens"]),
        ),
    )
    target = _make_latent(target_video, target_audio, nested_factory)
    validate_av_latent(
        target,
        timeline_origin_frame=int(layout["window_start_frame"]),
        name="allocated_window",
    )
    return target


def extract_av_span(
    latent: Mapping[str, Any],
    *,
    start_frame: int,
    frame_count: int,
    timeline_origin_frame: int = 0,
) -> dict[str, Any]:
    """Extract one synchronized video/audio span at exact token boundaries."""

    origin = int(timeline_origin_frame)
    start = int(start_frame)
    count = int(frame_count)
    video, audio, total_frames = validate_av_latent(
        latent,
        timeline_origin_frame=origin,
    )
    if start < 0 or count < 1:
        raise ValueError("AV span requires start_frame >= 0 and frame_count >= 1")
    end = start + count
    if end > total_frames:
        raise ValueError(f"AV latent has {total_frames} frames but the span ends at {end}")
    video_start = visual_token_boundary(start)
    video_end = visual_token_boundary(end)
    global_start = origin + start
    global_end = origin + end
    audio_start = h3_audio_token_boundary(global_start) - h3_audio_token_boundary(origin)
    audio_end = h3_audio_token_boundary(global_end) - h3_audio_token_boundary(origin)
    descriptor: dict[str, Any] = {
        "timeline_origin_frame": origin,
        "local_start_frame": start,
        "local_end_frame": end,
        "global_start_frame": global_start,
        "global_end_frame": global_end,
        "frame_count": count,
        "video_start_token": video_start,
        "video_end_token": video_end,
        "video_tokens": video_end - video_start,
        "audio_start_token": audio_start,
        "audio_end_token": audio_end,
        "audio_tokens": audio_end - audio_start,
        "video_spatial_shape": [int(video.shape[3]), int(video.shape[4])],
        "video_dtype": _dtype(video),
        "audio_dtype": _dtype(audio),
        "video_device": _device(video),
        "audio_device": _device(audio),
    }
    descriptor_hash = content_hash(descriptor)
    return {
        "schema": AV_SPAN_SCHEMA,
        "descriptor": descriptor,
        "descriptor_hash": descriptor_hash,
        "video": _clone(video[:, :, video_start:video_end]),
        "audio": _clone(audio[..., audio_start:audio_end]),
    }


def validate_av_span(span: Mapping[str, Any]) -> tuple[Any, Any, Mapping[str, Any]]:
    if not isinstance(span, Mapping) or span.get("schema") != AV_SPAN_SCHEMA:
        raise ValueError(f"AV span must use schema {AV_SPAN_SCHEMA}")
    descriptor = span.get("descriptor")
    if not isinstance(descriptor, Mapping):
        raise ValueError("AV span descriptor is missing")
    if span.get("descriptor_hash") != content_hash(descriptor):
        raise ValueError("AV span descriptor hash does not match its contents")
    video = span.get("video")
    audio = span.get("audio")
    if getattr(video, "ndim", 0) != 5 or getattr(audio, "ndim", 0) != 4:
        raise ValueError("AV span tensors have unexpected shapes")
    if int(video.shape[0]) != 1 or int(audio.shape[0]) != 1:
        raise ValueError("AV span video and audio batch sizes must both be 1")
    if int(video.shape[1]) != 24 or int(audio.shape[1]) != 32 or int(audio.shape[2]) != 2:
        raise ValueError("AV span tensors do not have MiniMax H3 channel geometry")
    fields = (
        "timeline_origin_frame",
        "local_start_frame",
        "local_end_frame",
        "global_start_frame",
        "global_end_frame",
        "frame_count",
        "video_start_token",
        "video_end_token",
        "video_tokens",
        "audio_start_token",
        "audio_end_token",
        "audio_tokens",
    )
    try:
        values = {field: int(descriptor[field]) for field in fields}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("AV span descriptor is missing an integer field") from exc
    origin = values["timeline_origin_frame"]
    start = values["local_start_frame"]
    end = values["local_end_frame"]
    count = values["frame_count"]
    if origin < 0 or start < 0 or count < 1 or end != start + count:
        raise ValueError("AV span descriptor has an invalid local frame range")
    expected = {
        "global_start_frame": origin + start,
        "global_end_frame": origin + end,
        "video_start_token": visual_token_boundary(start),
        "video_end_token": visual_token_boundary(end),
        "audio_start_token": h3_audio_token_boundary(origin + start)
        - h3_audio_token_boundary(origin),
        "audio_end_token": h3_audio_token_boundary(origin + end)
        - h3_audio_token_boundary(origin),
    }
    expected["video_tokens"] = expected["video_end_token"] - expected["video_start_token"]
    expected["audio_tokens"] = expected["audio_end_token"] - expected["audio_start_token"]
    for field, expected_value in expected.items():
        if values[field] != expected_value:
            raise ValueError(
                f"AV span descriptor {field} must be {expected_value}, got {values[field]}"
            )
    if int(video.shape[2]) != values["video_tokens"]:
        raise ValueError("AV span video tensor differs from its descriptor")
    if int(audio.shape[-1]) != values["audio_tokens"]:
        raise ValueError("AV span audio tensor differs from its descriptor")
    if list(descriptor.get("video_spatial_shape", ())) != [
        int(video.shape[3]),
        int(video.shape[4]),
    ]:
        raise ValueError("AV span video spatial shape differs from its descriptor")
    if descriptor.get("video_dtype") != _dtype(video) or descriptor.get("audio_dtype") != _dtype(
        audio
    ):
        raise ValueError("AV span tensor dtype differs from its descriptor")
    if descriptor.get("video_device") != _device(video) or descriptor.get(
        "audio_device"
    ) != _device(audio):
        raise ValueError("AV span tensor device differs from its descriptor")
    return video, audio, descriptor


def build_av_span_keyframes(
    existing_keyframes: Sequence[Mapping[str, Any]],
    span: Mapping[str, Any],
    target_av_latent: Mapping[str, Any],
    target_layout: Mapping[str, Any],
    *,
    target_frame_idx: int,
) -> list[dict[str, Any]]:
    """Return keyframe metadata with one compatible latent AV span inserted."""

    validate_av_window_layout(target_layout)
    target_origin = int(target_layout["window_start_frame"])
    target_video, target_audio, target_frames = validate_av_latent(
        target_av_latent,
        timeline_origin_frame=target_origin,
        name="target_av_latent",
    )
    span_video, span_audio, descriptor = validate_av_span(span)
    _validate_tensor_compatibility(
        target_video,
        target_audio,
        span_video,
        span_audio,
    )
    index = int(target_frame_idx)
    span_frames = int(descriptor["frame_count"])
    if index < 0 or index + span_frames > target_frames:
        raise ValueError("latent AV guide must fit completely inside the target window")
    target_global_start = target_origin + index
    expected_audio_tokens = h3_audio_token_boundary(
        target_global_start + span_frames
    ) - h3_audio_token_boundary(target_global_start)
    if int(span_audio.shape[-1]) != expected_audio_tokens:
        raise ValueError(
            "latent AV guide audio length is not aligned at the requested target frame"
        )

    keyframes = [dict(item) for item in existing_keyframes]
    for keyframe in keyframes:
        position = keyframe.get("resolved_frame_index")
        if not isinstance(position, (int, float)):
            raise ValueError("existing H3 guides must expose resolved_frame_index")
        if int(position) < 0 or int(position) >= target_frames:
            raise ValueError("existing H3 guide starts outside the target window")
    keyframes.append(
        {
            "resolved_frame_index": index,
            "latent": _clone(span_video),
            "audio_latent": _clone(span_audio),
        }
    )
    keyframes.sort(key=lambda item: int(item["resolved_frame_index"]))
    return keyframes


def place_av_span(
    target_av_latent: Mapping[str, Any],
    span: Mapping[str, Any],
    *,
    target_frame_idx: int,
    timeline_origin_frame: int = 0,
    nested_factory: NestedFactory | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy one exact native AV span into a target without assigning denoise policy.

    Placement may deliberately rebase a span onto another global frame. The two
    streams must still occupy the same number of visual and audio tokens at the
    requested target position; incompatible 24->40 Hz phases fail closed.
    """

    origin = int(timeline_origin_frame)
    target_video, target_audio, target_frames = validate_av_latent(
        target_av_latent,
        timeline_origin_frame=origin,
        name="target_av_latent",
    )
    span_video, span_audio, descriptor = validate_av_span(span)
    _validate_tensor_compatibility(
        target_video,
        target_audio,
        span_video,
        span_audio,
    )
    index = int(target_frame_idx)
    span_frames = int(descriptor["frame_count"])
    end = index + span_frames
    if index < 0 or end > target_frames:
        raise ValueError("native AV span must fit completely inside the target latent")

    video_start = visual_token_boundary(index)
    video_end = visual_token_boundary(end)
    audio_start = h3_audio_token_boundary(origin + index) - h3_audio_token_boundary(origin)
    audio_end = h3_audio_token_boundary(origin + end) - h3_audio_token_boundary(origin)
    if video_end - video_start != int(span_video.shape[2]):
        raise ValueError("native AV span video tokens do not align at the requested target frame")
    if audio_end - audio_start != int(span_audio.shape[-1]):
        raise ValueError(
            "native AV span audio tokens do not align at the requested target frame"
        )

    placed_video = _clone(target_video)
    placed_audio = _clone(target_audio)
    placed_video[:, :, video_start:video_end] = span_video
    placed_audio[..., audio_start:audio_end] = span_audio
    placed = _with_streams(
        target_av_latent,
        placed_video,
        placed_audio,
        nested_factory,
    )
    target_global_start = origin + index
    report: dict[str, Any] = {
        "schema": "cauce.h3-av-placement-report/1",
        "timeline_origin_frame": origin,
        "target_frame_range": [index, end],
        "target_global_range": [target_global_start, origin + end],
        "source_global_range": [
            int(descriptor["global_start_frame"]),
            int(descriptor["global_end_frame"]),
        ],
        "frame_count": span_frames,
        "video_token_range": [video_start, video_end],
        "audio_token_range": [audio_start, audio_end],
        "rebased": int(descriptor["global_start_frame"]) != target_global_start,
        "source_descriptor_hash": span["descriptor_hash"],
    }
    report["placement_hash"] = content_hash(report)
    return placed, report


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
        video_profile.append(
            strengths[1] + (strengths[0] - strengths[1]) * weight
        )

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
        audio_profile.append(
            strengths[3] + (strengths[2] - strengths[3]) * weight
        )

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
        proposed_video[:, :, token_index] = strengths[1] + (
            strengths[0] - strengths[1]
        ) * spatial

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
        "temporal_projection": (
            "static" if source_frames == 1 else "amax-per-h3-visual-token"
        ),
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
        offset_h:offset_h + int(video.shape[3]),
        offset_w:offset_w + int(video.shape[4]),
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
        offset_h:offset_h + int(video.shape[3]),
        offset_w:offset_w + int(video.shape[4]),
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


def replace_av_span(
    base_av_latent: Mapping[str, Any],
    replacement_span: Mapping[str, Any],
    *,
    timeline_origin_frame: int = 0,
    nested_factory: NestedFactory | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace one globally aligned native AV interval without changing duration."""

    origin = int(timeline_origin_frame)
    _, _, total_frames = validate_av_latent(
        base_av_latent,
        timeline_origin_frame=origin,
        name="base_av_latent",
    )
    _, _, descriptor = validate_av_span(replacement_span)
    target_index = int(descriptor["global_start_frame"]) - origin
    if target_index < 0 or int(descriptor["global_end_frame"]) > origin + total_frames:
        raise ValueError("replacement AV span lies outside the base latent timeline")
    replaced, placement = place_av_span(
        base_av_latent,
        replacement_span,
        target_frame_idx=target_index,
        timeline_origin_frame=origin,
        nested_factory=nested_factory,
    )
    if placement["rebased"]:
        raise ValueError("replacement AV span must retain its original global frame range")
    replaced.pop("noise_mask", None)
    report: dict[str, Any] = {
        "schema": "cauce.h3-av-replacement-report/1",
        "timeline_origin_frame": origin,
        "target_frame_count": total_frames,
        "replaced_global_range": placement["target_global_range"],
        "source_descriptor_hash": replacement_span["descriptor_hash"],
        "placement_hash": placement["placement_hash"],
    }
    report["replacement_hash"] = content_hash(report)
    return replaced, report


def append_av_span(
    base_av_latent: Mapping[str, Any],
    span: Mapping[str, Any],
    *,
    nested_factory: NestedFactory | None = None,
) -> tuple[dict[str, Any], int]:
    """Append one globally contiguous AV span without resampling or overlap policy."""

    base_video, base_audio, base_frames = validate_av_latent(
        base_av_latent,
        name="base_av_latent",
    )
    span_video, span_audio, descriptor = validate_av_span(span)
    if int(descriptor["global_start_frame"]) != base_frames:
        raise ValueError(
            "AV span is not globally contiguous with the base latent: "
            f"expected {base_frames}, got {descriptor['global_start_frame']}"
        )
    _validate_tensor_compatibility(base_video, base_audio, span_video, span_audio)
    extended_video = _concatenate((base_video, span_video), axis=2)
    extended_audio = _concatenate((base_audio, span_audio), axis=-1)
    extended = _make_latent(extended_video, extended_audio, nested_factory)
    total_frames = int(descriptor["global_end_frame"])
    _, _, validated_frames = validate_av_latent(extended, name="extended_av_latent")
    if validated_frames != total_frames:
        raise ValueError("appended AV latent length differs from the span timeline")
    return extended, total_frames


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
    generated_indices = [
        index for index in range(target_token_count) if index not in anchor_set
    ]
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
            values["anchor_denoise"]
            + (values["gap_denoise"] - values["anchor_denoise"]) * weight
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


def split_av_latent(
    latent: Mapping[str, Any],
    *,
    cut_frame: int,
    nested_factory: NestedFactory | None = None,
) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    """Split an origin-zero cumulative state into prefix latent and suffix span."""

    cut = int(cut_frame)
    video, audio, total_frames = validate_av_latent(latent)
    if not is_h3_frame_count(cut):
        raise ValueError("cut_frame must leave a complete 17k+5 H3 prefix")
    if cut >= total_frames:
        raise ValueError("cut_frame must leave a non-empty suffix")
    video_end = visual_token_boundary(cut)
    audio_end = h3_audio_token_boundary(cut)
    prefix = _make_latent(
        _clone(video[:, :, :video_end]),
        _clone(audio[..., :audio_end]),
        nested_factory,
    )
    _, _, prefix_frames = validate_av_latent(
        prefix,
        name="prefix_av_latent",
    )
    suffix_frames = total_frames - cut
    suffix = extract_av_span(
        latent,
        start_frame=cut,
        frame_count=suffix_frames,
    )
    return prefix, suffix, prefix_frames, suffix_frames
