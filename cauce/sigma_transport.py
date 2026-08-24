"""Sigma-conditioned operator splitting for native H3 latent transport.

This module wraps an existing deterministic ``res_multistep`` sampler rather
than reimplementing its solver.  Immediately before each model evaluation, a
small incremental pullback is applied to the packed H3 visual latent.  The
denoiser therefore observes the transported state while ComfyUI keeps the
solver, sigma schedule, conditioning, progress callback, and multistep history
inside one normal sampling call.

Only the visual stream is resampled.  The packed audio stream is unpacked and
repacked byte-for-byte without spatial transformation.
"""

from __future__ import annotations

import json
import math
from typing import Any

from .motion import PADDING_MODES, _h3_positions, _torch_grid, identity_grid, validate_motion_map


SIGMA_ENVELOPES = ("accumulate", "pulse")
SIGMA_EASINGS = ("linear", "smoothstep", "cosine")
SUPPORTED_SAMPLERS = ("sample_res_multistep", "sample_res_multistep_cfg_pp")


def _ease_unit(value: float, easing: str) -> float:
    x = min(1.0, max(0.0, float(value)))
    if easing == "linear":
        return x
    if easing == "smoothstep":
        return x * x * (3.0 - 2.0 * x)
    if easing == "cosine":
        return 0.5 - 0.5 * math.cos(math.pi * x)
    raise ValueError(f"sigma easing must be one of {', '.join(SIGMA_EASINGS)}")


def validate_sigma_schedule(
    start_percent: float,
    end_percent: float,
    strength: float,
    envelope: str,
    easing: str,
) -> None:
    if not 0.0 <= float(start_percent) < float(end_percent) <= 1.0:
        raise ValueError("sigma schedule requires 0 <= start < end <= 1")
    if not -1.0 <= float(strength) <= 1.0:
        raise ValueError("sigma transport strength must lie in [-1,1]")
    if envelope not in SIGMA_ENVELOPES:
        raise ValueError(f"sigma envelope must be one of {', '.join(SIGMA_ENVELOPES)}")
    if easing not in SIGMA_EASINGS:
        raise ValueError(f"sigma easing must be one of {', '.join(SIGMA_EASINGS)}")


def sigma_envelope_value(
    step_index: int,
    total_steps: int,
    *,
    start_percent: float,
    end_percent: float,
    strength: float,
    envelope: str = "accumulate",
    easing: str = "smoothstep",
) -> float:
    """Return the cumulative transport strength at one solver step.

    Percent zero is the highest-sigma model evaluation; percent one is the
    lowest-sigma evaluation. ``accumulate`` ramps to a retained transform.
    ``pulse`` rises to the midpoint and returns to identity before the end.
    """

    validate_sigma_schedule(start_percent, end_percent, strength, envelope, easing)
    if int(total_steps) < 1:
        raise ValueError("total sampler steps must be positive")
    if not 0 <= int(step_index) < int(total_steps):
        raise ValueError("step index lies outside the sampler schedule")
    progress = 1.0 if int(total_steps) == 1 else int(step_index) / (int(total_steps) - 1)
    start, end = float(start_percent), float(end_percent)
    amount = float(strength)
    if envelope == "accumulate":
        if progress <= start:
            return 0.0
        if progress >= end:
            return amount
        return amount * _ease_unit((progress - start) / (end - start), easing)

    midpoint = 0.5 * (start + end)
    if progress <= start or progress >= end:
        return 0.0
    if progress <= midpoint:
        return amount * _ease_unit((progress - start) / (midpoint - start), easing)
    return amount * _ease_unit((end - progress) / (end - midpoint), easing)


def sigma_schedule_series(total_steps: int, **parameters: Any) -> list[float]:
    return [
        sigma_envelope_value(index, total_steps, **parameters)
        for index in range(int(total_steps))
    ]


def sigma_schedule_increments(total_steps: int, **parameters: Any) -> list[float]:
    previous = 0.0
    increments = []
    for value in sigma_schedule_series(total_steps, **parameters):
        increments.append(value - previous)
        previous = value
    return increments


def warp_h3_video_step(
    video: Any,
    motion_map: dict[str, Any],
    incremental_strength: float,
    *,
    padding_mode: str = "reflection",
):
    """Apply one small pullback to a ``[B,C,T,H,W]`` H3 visual stream."""

    import torch
    import torch.nn.functional as functional

    validate_motion_map(motion_map)
    if padding_mode not in PADDING_MODES:
        raise ValueError(f"padding mode must be one of {', '.join(PADDING_MODES)}")
    if not isinstance(video, torch.Tensor) or video.ndim != 5:
        raise ValueError("H3 visual latent must have shape [batch,channels,tokens,height,width]")
    delta = float(incremental_strength)
    if abs(delta) <= 1e-12:
        return video

    batch, channels, tokens, height, width = map(int, video.shape)
    positions = _h3_positions(tokens)
    full_grid, _ = _torch_grid(
        motion_map,
        tokens,
        height,
        width,
        video.device,
        video.dtype,
        positions=positions,
    )
    identity = torch.from_numpy(identity_grid(tokens, height, width)).to(
        device=video.device, dtype=video.dtype
    )
    incremental_grid = identity + delta * (full_grid - identity)
    source = video.permute(0, 2, 1, 3, 4).reshape(
        batch * tokens, channels, height, width
    )
    repeated = incremental_grid.unsqueeze(0).expand(
        batch, -1, -1, -1, -1
    ).reshape(batch * tokens, height, width, 2)
    return functional.grid_sample(
        source,
        repeated,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=False,
    ).reshape(batch, tokens, channels, height, width).permute(0, 2, 1, 3, 4)


class _TransportModelProxy:
    def __init__(self, owner: "SigmaMotionSampler", wrapped: Any):
        self.owner = owner
        self.wrapped = wrapped

    def __getattr__(self, name: str):
        return getattr(self.wrapped, name)

    def __call__(self, x, sigma, *args, **kwargs):
        self.owner._transport_packed_state_in_place(x)
        return self.wrapped(x, sigma, *args, **kwargs)


class SigmaMotionSampler:
    """Comfy ``SAMPLER`` wrapper for first-order transport/diffusion splitting."""

    def __init__(
        self,
        base_sampler: Any,
        motion_map: dict[str, Any],
        *,
        start_percent: float = 0.1,
        end_percent: float = 0.65,
        strength: float = 0.25,
        envelope: str = "accumulate",
        easing: str = "smoothstep",
        padding_mode: str = "reflection",
    ):
        validate_motion_map(motion_map)
        validate_sigma_schedule(start_percent, end_percent, strength, envelope, easing)
        if padding_mode not in PADDING_MODES:
            raise ValueError(f"padding mode must be one of {', '.join(PADDING_MODES)}")
        function = getattr(base_sampler, "sampler_function", None)
        sampler_name = getattr(function, "__name__", "")
        if sampler_name not in SUPPORTED_SAMPLERS:
            raise ValueError(
                "sigma-conditioned transport currently supports deterministic "
                "res_multistep and res_multistep_cfg_pp only"
            )
        self.base_sampler = base_sampler
        self.motion_map = motion_map
        self.start_percent = float(start_percent)
        self.end_percent = float(end_percent)
        self.strength = float(strength)
        self.envelope = envelope
        self.easing = easing
        self.padding_mode = padding_mode
        self.sampler_name = sampler_name
        self._step_index = 0
        self._total_steps = 0
        self._previous_strength = 0.0
        self._latent_shapes: list[tuple[int, ...]] = []

    def report(self) -> dict[str, Any]:
        return {
            "schema": "cauce.sigma-conditioned-transport/1",
            "solver": self.sampler_name,
            "map_hash": self.motion_map["tensor_hash"],
            "operator_split": "transport_before_model_evaluation",
            "stream_policy": "warp_visual_copy_audio",
            "start_percent": self.start_percent,
            "end_percent": self.end_percent,
            "strength": self.strength,
            "envelope": self.envelope,
            "easing": self.easing,
            "padding_mode": self.padding_mode,
            "supported_samplers": list(SUPPORTED_SAMPLERS),
        }

    def _transport_packed_state_in_place(self, packed):
        import comfy.utils  # type: ignore

        if self._step_index >= self._total_steps:
            raise RuntimeError("sampler performed more model evaluations than sigma steps")
        cumulative = sigma_envelope_value(
            self._step_index,
            self._total_steps,
            start_percent=self.start_percent,
            end_percent=self.end_percent,
            strength=self.strength,
            envelope=self.envelope,
            easing=self.easing,
        )
        incremental = cumulative - self._previous_strength
        self._previous_strength = cumulative
        self._step_index += 1
        if abs(incremental) <= 1e-12:
            return

        streams = list(comfy.utils.unpack_latents(packed, self._latent_shapes))
        if len(streams) < 2 or streams[0].ndim != 5 or streams[1].ndim != 4:
            raise RuntimeError("sigma transport requires a packed MiniMax H3 AV latent")
        audio_before = streams[1]
        streams[0] = warp_h3_video_step(
            streams[0],
            self.motion_map,
            incremental,
            padding_mode=self.padding_mode,
        )
        streams[1] = audio_before
        transported, _ = comfy.utils.pack_latents(streams)
        packed.copy_(transported)

    def sample(
        self,
        model_wrap,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        total_steps = int(len(sigmas) - 1)
        if total_steps < 1:
            return latent_image if latent_image is not None else noise
        latent_shapes = getattr(model_wrap.inner_model, "latent_shapes", None)
        if not latent_shapes or len(latent_shapes) < 2:
            raise RuntimeError("sigma transport requires ComfyUI's packed H3 latent metadata")
        self._latent_shapes = [tuple(shape) for shape in latent_shapes]
        self._total_steps = total_steps
        self._step_index = 0
        self._previous_strength = 0.0
        proxy = _TransportModelProxy(self, model_wrap)
        result = self.base_sampler.sample(
            proxy,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )
        if self._step_index != total_steps:
            raise RuntimeError(
                f"expected {total_steps} model evaluations, observed {self._step_index}; "
                "refusing an ambiguous sigma schedule"
            )
        return result


def sigma_transport_report_json(sampler: SigmaMotionSampler) -> str:
    return json.dumps(sampler.report(), ensure_ascii=False, indent=2, sort_keys=True)
