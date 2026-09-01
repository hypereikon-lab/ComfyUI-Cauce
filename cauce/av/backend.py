"""One typed tensor boundary for the optional Torch and NumPy backends.

The operation modules address tensor behavior through this protocol.  Backend
detection is centralized here instead of being repeated throughout H3 math.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from ..h3 import get_av_streams
from .types import NestedFactory


@runtime_checkable
class TensorBackend(Protocol):
    name: str

    def accepts(self, value: Any) -> bool: ...
    def clone(self, value: Any) -> Any: ...
    def zeros(self, reference: Any, shape: tuple[int, ...]) -> Any: ...
    def full(self, reference: Any, shape: tuple[int, ...], value: float) -> Any: ...
    def concatenate(self, values: tuple[Any, ...], axis: int) -> Any: ...
    def profile(self, reference: Any, values: Sequence[float]) -> Any: ...
    def mask(self, reference: Any, value: Any) -> Any: ...


class _TorchBackend:
    name = "torch"

    @staticmethod
    def _torch():
        import torch

        return torch

    def accepts(self, value: Any) -> bool:
        try:
            return isinstance(value, self._torch().Tensor)
        except ImportError:
            return False

    def clone(self, value: Any) -> Any:
        return value.clone()

    def zeros(self, reference: Any, shape: tuple[int, ...]) -> Any:
        return reference.new_zeros(shape)

    def full(self, reference: Any, shape: tuple[int, ...], value: float) -> Any:
        return reference.new_full(shape, float(value))

    def concatenate(self, values: tuple[Any, ...], axis: int) -> Any:
        return self._torch().cat(values, dim=axis)

    def profile(self, reference: Any, values: Sequence[float]) -> Any:
        torch = self._torch()
        return torch.tensor(values, dtype=torch.float32, device=reference.device)

    def mask(self, reference: Any, value: Any) -> Any:
        torch = self._torch()
        if isinstance(value, torch.Tensor):
            return value.to(device=reference.device, dtype=torch.float32)
        return torch.as_tensor(value, device=reference.device, dtype=torch.float32)


class _NumpyBackend:
    name = "numpy"

    @staticmethod
    def _numpy():
        import numpy as np

        return np

    def accepts(self, value: Any) -> bool:
        try:
            return isinstance(value, self._numpy().ndarray)
        except ImportError:
            return False

    def clone(self, value: Any) -> Any:
        return value.copy()

    def zeros(self, reference: Any, shape: tuple[int, ...]) -> Any:
        return self._numpy().zeros(shape, dtype=reference.dtype)

    def full(self, reference: Any, shape: tuple[int, ...], value: float) -> Any:
        return self._numpy().full(shape, float(value), dtype=reference.dtype)

    def concatenate(self, values: tuple[Any, ...], axis: int) -> Any:
        return self._numpy().concatenate(values, axis=axis)

    def profile(self, reference: Any, values: Sequence[float]) -> Any:
        return self._numpy().asarray(values, dtype=self._numpy().float32)

    def mask(self, reference: Any, value: Any) -> Any:
        return self._numpy().asarray(value, dtype=self._numpy().float32)


_BACKENDS: tuple[TensorBackend, ...] = (_TorchBackend(), _NumpyBackend())


def backend_for(value: Any) -> TensorBackend:
    for backend in _BACKENDS:
        if backend.accepts(value):
            return backend
    raise TypeError("AV tensors must be PyTorch tensors or NumPy arrays")


def clone(value: Any) -> Any:
    return backend_for(value).clone(value)


def new_zeros(reference: Any, shape: tuple[int, ...]) -> Any:
    return backend_for(reference).zeros(reference, shape)


def new_full(reference: Any, shape: tuple[int, ...], value: float) -> Any:
    return backend_for(reference).full(reference, shape, value)


def concatenate(values: tuple[Any, ...], axis: int) -> Any:
    if not values:
        raise ValueError("cannot concatenate an empty tensor sequence")
    backend = backend_for(values[0])
    if any(backend_for(value).name != backend.name for value in values[1:]):
        raise TypeError("all concatenated AV tensors must share one backend")
    return backend.concatenate(values, axis)


def resize_spatial(value: Any, height: int, width: int, method: str) -> Any:
    target = (int(height), int(width))
    if min(target) < 1:
        raise ValueError("spatial target dimensions must be positive")
    if tuple(int(item) for item in value.shape[-2:]) == target:
        return clone(value)
    modes = {"nearest-exact", "bilinear", "bicubic", "area"}
    if method not in modes:
        raise ValueError(f"resize method must be one of {sorted(modes)}")
    backend = backend_for(value)
    if backend.name == "torch":
        from torch.nn import functional

        source_shape = tuple(int(item) for item in value.shape)
        flat = value.reshape((-1, 1, source_shape[-2], source_shape[-1]))
        kwargs: dict[str, Any] = {}
        if method in {"bilinear", "bicubic"}:
            kwargs["align_corners"] = False
        resized = functional.interpolate(flat, size=target, mode=method, **kwargs)
        return resized.reshape(source_shape[:-2] + target)
    np = _NumpyBackend._numpy()
    ys = np.rint(np.linspace(0, value.shape[-2] - 1, target[0])).astype(np.int64)
    xs = np.rint(np.linspace(0, value.shape[-1] - 1, target[1])).astype(np.int64)
    return np.take(np.take(value, ys, axis=-2), xs, axis=-1)


def make_latent(video: Any, audio: Any, nested_factory: NestedFactory | None) -> dict[str, Any]:
    samples = nested_factory((video, audio)) if nested_factory is not None else (video, audio)
    return {"samples": samples}


def with_streams(
    latent: dict[str, Any] | Any,
    video: Any,
    audio: Any,
    nested_factory: NestedFactory | None,
) -> dict[str, Any]:
    out = dict(latent)
    out["samples"] = (
        nested_factory((video, audio)) if nested_factory is not None else (video, audio)
    )
    return out


def profile_tensor(reference: Any, values: Sequence[float]) -> Any:
    return backend_for(reference).profile(reference, values)


def mask_tensor(reference: Any, value: Any) -> Any:
    return backend_for(reference).mask(reference, value)


def mask_min_max(value: Any) -> tuple[float, float]:
    try:
        return float(value.min().item()), float(value.max().item())
    except AttributeError:
        return float(value.min()), float(value.max())


def mask_all_finite(value: Any) -> bool:
    backend = backend_for(value)
    if backend.name == "torch":
        import torch

        return bool(torch.isfinite(value).all().item())
    np = _NumpyBackend._numpy()
    return bool(np.isfinite(value).all())


def mask_digest(value: Any) -> str:
    backend = backend_for(value)
    if backend.name == "torch":
        import torch

        array = value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    else:
        array = value
    np = _NumpyBackend._numpy()
    canonical = np.asarray(array, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(tuple(int(item) for item in canonical.shape)).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def resize_mask_frames(mask: Any, height: int, width: int) -> Any:
    target = (int(height), int(width))
    if tuple(int(item) for item in mask.shape[-2:]) == target:
        return clone(mask)
    backend = backend_for(mask)
    if backend.name == "torch":
        from torch.nn import functional

        return functional.interpolate(
            mask.unsqueeze(1), size=target, mode="bilinear", align_corners=False
        ).squeeze(1)
    np = _NumpyBackend._numpy()
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


def temporal_amax(value: Any, start: int, end: int) -> Any:
    segment = value[int(start) : int(end)]
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


def broadcast_video_mask(profile: Any, video: Any) -> Any:
    shape = (int(video.shape[0]), 1, int(video.shape[2]), int(video.shape[3]), int(video.shape[4]))
    backend = backend_for(profile)
    if backend.name == "torch":
        return profile.reshape(1, 1, -1, 1, 1).expand(shape).clone()
    return _NumpyBackend._numpy().broadcast_to(profile.reshape(1, 1, -1, 1, 1), shape).copy()


def broadcast_audio_mask(profile: Any, audio: Any) -> Any:
    shape = (int(audio.shape[0]), 1, int(audio.shape[2]), int(audio.shape[3]))
    backend = backend_for(profile)
    if backend.name == "torch":
        return profile.reshape(1, 1, 1, -1).expand(shape).clone()
    return _NumpyBackend._numpy().broadcast_to(profile.reshape(1, 1, 1, -1), shape).copy()


def combine_mask(existing: Any, proposed: Any, mode: str) -> Any:
    if mode == "replace" or existing is None:
        return proposed
    if tuple(existing.shape) != tuple(proposed.shape):
        raise ValueError("existing AV noise_mask shape differs from the proposed mask")
    if mode == "maximum":
        maximum = getattr(existing, "maximum", None)
        return (
            maximum(proposed)
            if callable(maximum)
            else _NumpyBackend._numpy().maximum(existing, proposed)
        )
    if mode == "minimum":
        minimum = getattr(existing, "minimum", None)
        return (
            minimum(proposed)
            if callable(minimum)
            else _NumpyBackend._numpy().minimum(existing, proposed)
        )
    if mode == "multiply":
        return existing * proposed
    raise ValueError("mask combine mode must be replace, maximum, minimum, or multiply")


def mask_streams(latent: Any) -> tuple[Any | None, Any | None]:
    value = latent.get("noise_mask")
    if value is None:
        return None, None
    try:
        return get_av_streams({"samples": value})
    except (TypeError, ValueError) as exc:
        raise ValueError("H3 noise_mask must contain nested video and audio streams") from exc


def curve(value: float, name: str) -> float:
    x = min(1.0, max(0.0, float(value)))
    if name == "linear":
        return x
    if name == "smoothstep":
        return x * x * (3.0 - 2.0 * x)
    if name == "smootherstep":
        return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
    raise ValueError("mask curve must be linear, smoothstep, or smootherstep")


def interval_weight(
    position: float, start: int, end: int, fade_in: int, fade_out: int, name: str
) -> float:
    if start <= position < end:
        return 1.0
    if fade_in > 0 and start - fade_in < position < start:
        return curve((position - (start - fade_in)) / fade_in, name)
    if fade_out > 0 and end <= position < end + fade_out:
        return curve(1.0 - ((position - end) / fade_out), name)
    return 0.0


def dtype_name(value: Any) -> str:
    return str(getattr(value, "dtype", "unknown"))


def device_name(value: Any) -> str:
    return str(getattr(value, "device", "cpu"))


def validate_tensor_compatibility(
    left_video: Any, left_audio: Any, right_video: Any, right_audio: Any
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
    if device_name(left_video) != device_name(right_video):
        raise ValueError("H3 video devices must match")
    if device_name(left_audio) != device_name(right_audio):
        raise ValueError("H3 structural-audio devices must match")
