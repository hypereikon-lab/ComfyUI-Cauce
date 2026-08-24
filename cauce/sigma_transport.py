"""Sigma-conditioned operator splitting for native H3 latent transport.

This module preserves ComfyUI's deterministic ``res_multistep`` equations and
adds one covariant operator-splitting step.  Immediately before each model
evaluation, the same small incremental pullback is applied to both the packed
H3 state and the solver's retained denoised history.  Keeping both tensors in
the same coordinate frame is essential: transporting only the current state
causes the second-order residual to compare misregistered fields and produces
strong banding artifacts.

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
            "schema": "cauce.sigma-conditioned-transport/2",
            "solver": self.sampler_name,
            "map_hash": self.motion_map["tensor_hash"],
            "operator_split": "covariant_transport_of_state_and_solver_history",
            "stream_policy": "warp_visual_copy_audio",
            "start_percent": self.start_percent,
            "end_percent": self.end_percent,
            "strength": self.strength,
            "envelope": self.envelope,
            "easing": self.easing,
            "padding_mode": self.padding_mode,
            "supported_samplers": list(SUPPORTED_SAMPLERS),
        }

    def _transport_packed_state(self, packed, incremental: float):
        import comfy.utils  # type: ignore

        if abs(incremental) <= 1e-12:
            return packed

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
        return transported

    def _next_increment(self) -> float:
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
        return incremental

    def _integrated_res_multistep(
        self,
        model,
        x,
        sigmas,
        extra_args=None,
        callback=None,
        disable=None,
        s_noise=1.0,
        noise_sampler=None,
    ):
        """Comfy's deterministic RES solver with covariant history transport.

        The numerical update intentionally follows ComfyUI's ``res_multistep``
        implementation.  The CAUCE addition is confined to the start of each
        iteration, where ``x`` and ``old_denoised`` receive the same incremental
        pullback before the next prediction is evaluated.
        """

        import torch
        import comfy.k_diffusion.sampling as sampling  # type: ignore
        import comfy.model_patcher  # type: ignore

        extra_args = {} if extra_args is None else extra_args.copy()
        seed = extra_args.get("seed", None)
        noise_sampler = (
            sampling.default_noise_sampler(x, seed=seed)
            if noise_sampler is None
            else noise_sampler
        )
        noise_scale = getattr(
            model.inner_model.model_patcher.get_model_object("model_sampling"),
            "noise_scale",
            1.0,
        )
        s_noise = float(s_noise) * noise_scale
        s_in = x.new_ones([x.shape[0]])
        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()
        phi1_fn = lambda t: torch.expm1(t) / t
        phi2_fn = lambda t: (phi1_fn(t) - 1.0) / t

        old_sigma_down = None
        old_denoised = None
        uncond_denoised = None
        cfg_pp = self.sampler_name == "sample_res_multistep_cfg_pp"

        def post_cfg_function(args):
            nonlocal uncond_denoised
            uncond_denoised = args["uncond_denoised"]
            return args["denoised"]

        if cfg_pp:
            model_options = extra_args.get("model_options", {}).copy()
            extra_args["model_options"] = (
                comfy.model_patcher.set_model_options_post_cfg_function(
                    model_options,
                    post_cfg_function,
                    disable_cfg1_optimization=True,
                )
            )

        for index in range(len(sigmas) - 1):
            incremental = self._next_increment()
            if abs(incremental) > 1e-12:
                x = self._transport_packed_state(x, incremental)
                if old_denoised is not None:
                    old_denoised = self._transport_packed_state(
                        old_denoised, incremental
                    )

            denoised = model(x, sigmas[index] * s_in, **extra_args)
            sigma_down, sigma_up = sampling.get_ancestral_step(
                sigmas[index], sigmas[index + 1], eta=0.0
            )
            if callback is not None:
                callback(
                    {
                        "x": x,
                        "i": index,
                        "sigma": sigmas[index],
                        "sigma_hat": sigmas[index],
                        "denoised": denoised,
                    }
                )

            if sigma_down == 0 or old_denoised is None:
                if cfg_pp:
                    derivative = sampling.to_d(x, sigmas[index], uncond_denoised)
                    x = denoised + derivative * sigma_down
                else:
                    derivative = sampling.to_d(x, sigmas[index], denoised)
                    x = x + derivative * (sigma_down - sigmas[index])
            else:
                t = t_fn(sigmas[index])
                t_old = t_fn(old_sigma_down)
                t_next = t_fn(sigma_down)
                t_prev = t_fn(sigmas[index - 1])
                h = t_next - t
                c2 = (t_prev - t_old) / h
                phi1_value, phi2_value = phi1_fn(-h), phi2_fn(-h)
                b1 = torch.nan_to_num(phi1_value - phi2_value / c2, nan=0.0)
                b2 = torch.nan_to_num(phi2_value / c2, nan=0.0)
                if cfg_pp:
                    x = x + (denoised - uncond_denoised)
                    x = sigma_fn(h) * x + h * (
                        b1 * uncond_denoised + b2 * old_denoised
                    )
                else:
                    x = sigma_fn(h) * x + h * (
                        b1 * denoised + b2 * old_denoised
                    )

            if sigma_up > 0:
                x = x + (
                    noise_sampler(sigmas[index], sigmas[index + 1])
                    * s_noise
                    * sigma_up
                )

            old_denoised = uncond_denoised if cfg_pp else denoised
            old_sigma_down = sigma_down
        return x

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
        import comfy.samplers  # type: ignore

        integrated_sampler = comfy.samplers.KSAMPLER(
            self._integrated_res_multistep,
            extra_options=dict(getattr(self.base_sampler, "extra_options", {})),
            inpaint_options=dict(getattr(self.base_sampler, "inpaint_options", {})),
        )
        result = integrated_sampler.sample(
            model_wrap,
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
