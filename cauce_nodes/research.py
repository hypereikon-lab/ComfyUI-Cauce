"""Explicitly experimental ComfyUI bindings with no production guarantee."""

from __future__ import annotations

import json

from ..cauce.motion import (
    PADDING_MODES,
    WarpedH3Noise,
    motion_map_report,
    warp_h3_latent,
)
from ..cauce.latent_injection import (
    H3FlowLatentInjectionSampler,
    MASK_PROJECTIONS,
    flow_injection_report_json,
)
from ..cauce.seams import (
    NATIVE_SEAM_CONTEXT_FRAMES,
    NATIVE_SEAM_WORKING_FRAMES,
    build_native_latent_seam_window,
    make_native_latent_seam_plan,
    prepare_h3_native_latent_temporal_inpaint,
)
from ..cauce.sigma_transport import (
    SIGMA_EASINGS,
    SIGMA_ENVELOPES,
    SigmaMotionSampler,
    sigma_transport_report_json,
)


CATEGORY = "CAUCE/Research"


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


class CauceBuildNativeLatentSeam:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left_frames": ("IMAGE",),
                "right_frames": ("IMAGE",),
                "left_fps": ("FLOAT", {"default": 24.0}),
                "right_fps": ("FLOAT", {"default": 24.0}),
                "context_frames_per_side": (
                    [str(value) for value in NATIVE_SEAM_CONTEXT_FRAMES],
                    {"default": "22"},
                ),
                "working_frames": (
                    [str(value) for value in NATIVE_SEAM_WORKING_FRAMES],
                    {"default": "124"},
                ),
                "accepted_repair_frames": (
                    "INT",
                    {"default": 72, "min": 2, "max": 358, "step": 2},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "CAUCE_SEAM", "STRING")
    RETURN_NAMES = ("working_images", "seam", "seam_json")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Research: plan a bidirectional seam whose protected context comes from "
        "phase-matched source H3 latents."
    )

    def build(
        self,
        left_frames,
        right_frames,
        left_fps,
        right_fps,
        context_frames_per_side,
        working_frames,
        accepted_repair_frames,
    ):
        if abs(float(left_fps) - 24.0) > 1e-3 or abs(float(right_fps) - 24.0) > 1e-3:
            raise ValueError("native H3 temporal inpainting requires 24 fps sources")
        plan = make_native_latent_seam_plan(
            int(left_frames.shape[0]),
            int(right_frames.shape[0]),
            context_frames_per_side=int(context_frames_per_side),
            working_frames=int(working_frames),
            accepted_repair_frames=int(accepted_repair_frames),
        )
        working = build_native_latent_seam_window(left_frames, right_frames, plan)
        return working, plan, _json(plan)


class CaucePrepareH3NativeLatentInpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_latent": ("LATENT",),
                "left_latent": ("LATENT",),
                "right_latent": ("LATENT",),
                "seam": ("CAUCE_SEAM",),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("masked_latent", "mask_report")
    FUNCTION = "prepare"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Research: copy phase-matched native H3 context into both sides of a target "
        "latent and expose only its center to denoising."
    )

    def prepare(self, target_latent, left_latent, right_latent, seam):
        latent, report = prepare_h3_native_latent_temporal_inpaint(
            target_latent, left_latent, right_latent, seam
        )
        return latent, _json(report)


class CauceWarpH3Latent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "motion_map": ("CAUCE_MAP",),
                "padding_mode": (list(PADDING_MODES), {"default": "border"}),
                "mask_mode": (["none", "holes", "all"], {"default": "holes"}),
            }
        }

    RETURN_TYPES = ("LATENT", "MASK", "STRING")
    RETURN_NAMES = ("latent", "validity", "report_json")
    FUNCTION = "warp"
    CATEGORY = CATEGORY
    DESCRIPTION = "Research: apply a coordinate pullback directly to H3 visual latents."

    def warp(self, latent, motion_map, padding_mode, mask_mode):
        result, validity, report = warp_h3_latent(
            latent,
            motion_map,
            padding_mode=padding_mode,
            mask_mode=mask_mode,
        )
        return result, validity, _json(report)


class CauceWarpedH3Noise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "motion_map": ("CAUCE_MAP",),
                "padding_mode": (list(PADDING_MODES), {"default": "reflection"}),
                "temporal_correlation": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.005},
                ),
            }
        }

    RETURN_TYPES = ("NOISE", "STRING")
    RETURN_NAMES = ("noise", "report_json")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "Research: create weakly motion-correlated H3 visual noise."

    def build(self, seed, motion_map, padding_mode, temporal_correlation):
        report = motion_map_report(motion_map) | {
            "seed": int(seed),
            "padding_mode": padding_mode,
            "temporal_correlation": float(temporal_correlation),
            "output": "h3_warped_noise",
        }
        return (
            WarpedH3Noise(
                seed,
                motion_map,
                padding_mode,
                temporal_correlation=temporal_correlation,
            ),
            _json(report),
        )


class CauceSigmaMotionSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_sampler": ("SAMPLER",),
                "motion_map": ("CAUCE_MAP",),
                "start_percent": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 0.99, "step": 0.01},
                ),
                "end_percent": (
                    "FLOAT",
                    {"default": 0.65, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
                "strength": (
                    "FLOAT",
                    {"default": 0.1, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "envelope": (list(SIGMA_ENVELOPES), {"default": "accumulate"}),
                "easing": (list(SIGMA_EASINGS), {"default": "smoothstep"}),
                "padding_mode": (list(PADDING_MODES), {"default": "border"}),
            }
        }

    RETURN_TYPES = ("SAMPLER", "STRING")
    RETURN_NAMES = ("sampler", "report_json")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Research: transport H3 visual latents inside deterministic RES or Euler "
        "sampling while leaving structural audio unchanged."
    )

    def build(
        self,
        base_sampler,
        motion_map,
        start_percent,
        end_percent,
        strength,
        envelope,
        easing,
        padding_mode,
    ):
        sampler = SigmaMotionSampler(
            base_sampler,
            motion_map,
            start_percent=start_percent,
            end_percent=end_percent,
            strength=strength,
            envelope=envelope,
            easing=easing,
            padding_mode=padding_mode,
        )
        return sampler, sigma_transport_report_json(sampler)


class CauceH3FlowLatentInjectionSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "guide_latent": ("LATENT",),
                "flow_progress": (
                    "FLOAT",
                    {
                        "default": 0.45,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Target clean weight 1-sigma_next, not a linear step index.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "mask_projection": (
                    list(MASK_PROJECTIONS),
                    {"default": "mean"},
                ),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("SAMPLER", "STRING")
    RETURN_NAMES = ("sampler", "report_json")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Research: once during deterministic Euler flow sampling, partially "
        "substitute a same-geometry H3 visual clean estimate while preserving "
        "the implied noise endpoint and leaving structural audio unchanged."
    )

    def build(
        self,
        base_sampler,
        sigmas,
        guide_latent,
        flow_progress,
        strength,
        mask_projection,
        mask=None,
    ):
        sampler = H3FlowLatentInjectionSampler(
            base_sampler,
            sigmas,
            guide_latent,
            flow_progress=flow_progress,
            strength=strength,
            mask=mask,
            mask_projection=mask_projection,
        )
        return sampler, flow_injection_report_json(sampler)


NODE_CLASS_MAPPINGS = {
    "CauceBuildNativeLatentSeam": CauceBuildNativeLatentSeam,
    "CaucePrepareH3NativeLatentInpaint": CaucePrepareH3NativeLatentInpaint,
    "CauceWarpH3Latent": CauceWarpH3Latent,
    "CauceWarpedH3Noise": CauceWarpedH3Noise,
    "CauceSigmaMotionSampler": CauceSigmaMotionSampler,
    "CauceH3FlowLatentInjectionSampler": CauceH3FlowLatentInjectionSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceBuildNativeLatentSeam": "CAUCE Research · Build Native Latent Seam",
    "CaucePrepareH3NativeLatentInpaint": "CAUCE Research · Native Latent Inpaint",
    "CauceWarpH3Latent": "CAUCE Research · Warp H3 Latent",
    "CauceWarpedH3Noise": "CAUCE Research · Warped H3 Noise",
    "CauceSigmaMotionSampler": "CAUCE Research · Sigma Motion Sampler",
    "CauceH3FlowLatentInjectionSampler": (
        "CAUCE Research · H3 Flow Latent Injection Sampler"
    ),
}
