"""Experimental clean-estimate injection for native MiniMax H3 flow sampling.

The operation is deliberately narrower than a generic image-injection system.
It accepts a same-geometry H3 visual latent and replaces part of the current
clean-video estimate once during deterministic Euler sampling.  The current
flow residual is retained, and the packed structural-audio stream is copied
without modification.

For H3's rectified-flow state

``x_sigma = sigma * epsilon + (1 - sigma) * x0``

the corrected state at the same sigma is

``x'_sigma = x_sigma + a * M * (1 - sigma) * (guide - x0_hat)``.

At ``a*M = 1`` this substitutes the guide for the current clean estimate while
preserving the implied noise residual.  Fractional masks and strengths perform
a local interpolation.  A later model evaluation is always required so H3 can
project the intervention back toward its learned audiovisual manifold.
"""

from __future__ import annotations

import json
import math
from typing import Any

from .timebase import visual_span_for_tokens, visual_token_spans


SUPPORTED_INJECTION_SAMPLERS = ("sample_euler",)
MASK_PROJECTIONS = ("mean", "maximum")


def validate_flow_injection(inject_percent: float, strength: float) -> None:
    percent = float(inject_percent)
    amount = float(strength)
    if not math.isfinite(percent) or not 0.0 <= percent <= 1.0:
        raise ValueError("inject_percent must lie in [0,1]")
    if not math.isfinite(amount) or not 0.0 <= amount <= 1.0:
        raise ValueError("injection strength must lie in [0,1]")


def resolve_injection_step(total_steps: int, inject_percent: float) -> int:
    """Resolve the one Euler transition after which injection occurs.

    The final transition is excluded because an intervention at sigma zero
    would have no subsequent H3 evaluation to repair it.
    """

    validate_flow_injection(inject_percent, 0.0)
    count = int(total_steps)
    if count < 2:
        raise ValueError("H3 latent injection requires at least two sampler steps")
    requested = int(round(float(inject_percent) * float(count - 1)))
    return min(count - 2, max(0, requested))


def _video_samples(latent: dict[str, Any]):
    samples = latent.get("samples")
    if samples is None:
        raise ValueError("injection latent has no samples")
    if getattr(samples, "is_nested", False):
        streams = list(samples.unbind())
        if not streams:
            raise ValueError("injection H3 latent is empty")
        samples = streams[0]
    elif isinstance(samples, (list, tuple)):
        if not samples:
            raise ValueError("injection latent stream list is empty")
        samples = samples[0]
    if getattr(samples, "ndim", 0) != 5:
        raise ValueError(
            "injection latent must contain H3 video samples [B,C,T,H,W]"
        )
    return samples


def project_visible_mask_to_h3(
    mask: Any,
    *,
    tokens: int,
    height: int,
    width: int,
    projection: str = "mean",
):
    """Project a standard visible-frame ``MASK`` onto H3 visual tokens."""

    import torch
    import torch.nn.functional as functional

    if projection not in MASK_PROJECTIONS:
        raise ValueError(
            f"mask projection must be one of {', '.join(MASK_PROJECTIONS)}"
        )
    token_count = int(tokens)
    frames = visual_span_for_tokens(token_count)
    value = torch.as_tensor(mask, dtype=torch.float32)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim == 4 and int(value.shape[1]) == 1:
        value = value[:, 0]
    if value.ndim != 3:
        raise ValueError("injection mask must have shape [frames,height,width]")
    if int(value.shape[0]) not in {1, frames}:
        raise ValueError(
            f"injection mask has {value.shape[0]} frames; expected 1 or {frames}"
        )
    value = value.clamp(0.0, 1.0)
    if tuple(value.shape[-2:]) != (int(height), int(width)):
        value = functional.interpolate(
            value.unsqueeze(1),
            size=(int(height), int(width)),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    if int(value.shape[0]) == 1:
        value = value.expand(frames, -1, -1)

    reduced = []
    for start, end in visual_token_spans(token_count):
        interval = value[start:end]
        reduced.append(
            interval.amax(dim=0)
            if projection == "maximum"
            else interval.mean(dim=0)
        )
    return torch.stack(reduced, dim=0).unsqueeze(0).unsqueeze(0)


def flow_preserving_video_injection(
    current: Any,
    denoised: Any,
    guide: Any,
    mask: Any,
    *,
    sigma: float,
    strength: float,
):
    """Replace a clean estimate while retaining its rectified-flow residual."""

    import torch

    validate_flow_injection(0.0, strength)
    if not all(
        isinstance(value, torch.Tensor) for value in (current, denoised, guide)
    ):
        raise TypeError("flow injection expects tensor video streams")
    if (
        tuple(current.shape) != tuple(denoised.shape)
        or tuple(current.shape) != tuple(guide.shape)
    ):
        raise ValueError("current, denoised, and guide video shapes must match")
    if current.ndim != 5:
        raise ValueError("H3 visual streams must have shape [B,C,T,H,W]")
    sigma_value = float(sigma)
    if not math.isfinite(sigma_value) or not 0.0 <= sigma_value <= 1.0 + 1e-6:
        raise ValueError("H3 flow sigma must lie in [0,1]")
    weight = torch.as_tensor(mask, device=current.device, dtype=current.dtype)
    try:
        torch.broadcast_shapes(tuple(current.shape), tuple(weight.shape))
    except RuntimeError as exc:
        raise ValueError("injection mask is not broadcastable to H3 video") from exc
    amount = float(strength) * max(0.0, 1.0 - sigma_value)
    if amount <= 0.0:
        return current
    guide = guide.to(device=current.device, dtype=current.dtype)
    denoised = denoised.to(device=current.device, dtype=current.dtype)
    return current + amount * weight * (guide - denoised)


class H3FlowLatentInjectionSampler:
    """One-shot visual clean-estimate injection inside Euler flow sampling."""

    def __init__(
        self,
        base_sampler: Any,
        guide_latent: dict[str, Any],
        *,
        inject_percent: float = 0.45,
        strength: float = 0.15,
        mask: Any = None,
        mask_projection: str = "mean",
    ):
        validate_flow_injection(inject_percent, strength)
        if mask_projection not in MASK_PROJECTIONS:
            raise ValueError(
                f"mask projection must be one of {', '.join(MASK_PROJECTIONS)}"
            )
        function = getattr(base_sampler, "sampler_function", None)
        sampler_name = getattr(function, "__name__", "")
        if sampler_name not in SUPPORTED_INJECTION_SAMPLERS:
            raise ValueError(
                "H3 latent injection currently supports deterministic Euler only"
            )
        self.base_sampler = base_sampler
        self.guide_video = _video_samples(guide_latent)
        self.inject_percent = float(inject_percent)
        self.strength = float(strength)
        self.mask = mask
        self.mask_projection = mask_projection
        self.sampler_name = sampler_name
        self._latent_shapes: list[tuple[int, ...]] = []
        self._guide_internal = None
        self._mask_internal = None
        self._injection_step = -1
        self._observed_injections = 0

    def report(self) -> dict[str, Any]:
        return {
            "schema": "cauce.h3-flow-latent-injection/1",
            "status": "research",
            "solver": self.sampler_name,
            "operator": "one_shot_clean_estimate_substitution",
            "equation": "x'=x+a*M*(1-sigma)*(guide-x0_hat)",
            "inject_percent": self.inject_percent,
            "strength": self.strength,
            "mask_projection": self.mask_projection,
            "mask": "connected" if self.mask is not None else "full_visual_stream",
            "stream_policy": "inject_visual_copy_structural_audio",
            "history_policy": "euler_no_multistep_history",
            "post_injection_requirement": "at_least_one_model_evaluation",
        }

    def _prepare_runtime(self, model_wrap: Any, total_steps: int) -> None:
        import torch

        shapes = getattr(model_wrap.inner_model, "latent_shapes", None)
        if not shapes or len(shapes) < 2:
            raise RuntimeError(
                "H3 latent injection requires ComfyUI's packed AV latent metadata"
            )
        self._latent_shapes = [tuple(shape) for shape in shapes]
        target_shape = self._latent_shapes[0]
        if len(target_shape) != 5 or len(self._latent_shapes[1]) != 4:
            raise RuntimeError(
                "H3 latent injection received unexpected AV stream shapes"
            )
        if tuple(self.guide_video.shape) != target_shape:
            raise ValueError(
                "guide video latent geometry does not match the sampled H3 target: "
                f"{tuple(self.guide_video.shape)} != {target_shape}"
            )
        guide = self.guide_video.detach().clone()
        # A guide is a visual stream, not a packed AV latent. Calling H3's
        # process_latent_in here would also enter its packed-audio scaling path.
        guide = model_wrap.inner_model.latent_format.process_in(guide)
        self._guide_internal = guide
        if self.mask is None:
            self._mask_internal = torch.ones(
                (1, 1, target_shape[2], target_shape[3], target_shape[4]),
                dtype=torch.float32,
            )
        else:
            self._mask_internal = project_visible_mask_to_h3(
                self.mask,
                tokens=target_shape[2],
                height=target_shape[3],
                width=target_shape[4],
                projection=self.mask_projection,
            ).detach()
        self._injection_step = resolve_injection_step(total_steps, self.inject_percent)
        self._observed_injections = 0

    def _inject_packed(self, current: Any, denoised: Any, sigma: float):
        import comfy.utils  # type: ignore

        current_streams = list(comfy.utils.unpack_latents(current, self._latent_shapes))
        clean_streams = list(comfy.utils.unpack_latents(denoised, self._latent_shapes))
        if len(current_streams) < 2 or len(clean_streams) < 2:
            raise RuntimeError("H3 packed state lost one of its AV streams")
        audio = current_streams[1]
        current_streams[0] = flow_preserving_video_injection(
            current_streams[0],
            clean_streams[0],
            self._guide_internal,
            self._mask_internal,
            sigma=float(sigma),
            strength=self.strength,
        )
        current_streams[1] = audio
        packed, _ = comfy.utils.pack_latents(current_streams)
        self._observed_injections += 1
        return packed

    def _integrated_euler_impl(
        self,
        model,
        x,
        sigmas,
        extra_args=None,
        callback=None,
        disable=None,
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
    ):
        import torch
        import comfy.k_diffusion.sampling as sampling  # type: ignore

        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]])
        for index in range(len(sigmas) - 1):
            if float(s_churn) > 0.0:
                gamma = (
                    min(float(s_churn) / (len(sigmas) - 1), 2**0.5 - 1.0)
                    if float(s_tmin) <= sigmas[index] <= float(s_tmax)
                    else 0.0
                )
            else:
                gamma = 0.0
            sigma_hat = sigmas[index] * (gamma + 1.0)
            if gamma > 0.0:
                epsilon = torch.randn_like(x) * float(s_noise)
                x = x + epsilon * (sigma_hat**2 - sigmas[index] ** 2) ** 0.5

            denoised = model(x, sigma_hat * s_in, **extra_args)
            derivative = sampling.to_d(x, sigma_hat, denoised)
            if callback is not None:
                callback(
                    {
                        "x": x,
                        "i": index,
                        "sigma": sigmas[index],
                        "sigma_hat": sigma_hat,
                        "denoised": denoised,
                    }
                )
            x = x + derivative * (sigmas[index + 1] - sigma_hat)
            if index == self._injection_step and self.strength > 0.0:
                x = self._inject_packed(x, denoised, float(sigmas[index + 1]))
        return x

    def _integrated_euler(
        self,
        model,
        x,
        sigmas,
        extra_args=None,
        callback=None,
        disable=None,
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
    ):
        import torch

        with torch.no_grad():
            return self._integrated_euler_impl(
                model,
                x,
                sigmas,
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
            )

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
        self._prepare_runtime(model_wrap, total_steps)
        import comfy.samplers  # type: ignore

        integrated = comfy.samplers.KSAMPLER(
            self._integrated_euler,
            extra_options=dict(getattr(self.base_sampler, "extra_options", {})),
            inpaint_options=dict(getattr(self.base_sampler, "inpaint_options", {})),
        )
        result = integrated.sample(
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image,
            denoise_mask,
            disable_pbar,
        )
        expected = 0 if self.strength <= 0.0 else 1
        if self._observed_injections != expected:
            raise RuntimeError(
                f"expected {expected} H3 latent injection(s), observed "
                f"{self._observed_injections}"
            )
        return result


def flow_injection_report_json(sampler: H3FlowLatentInjectionSampler) -> str:
    return json.dumps(sampler.report(), ensure_ascii=False, indent=2, sort_keys=True)
