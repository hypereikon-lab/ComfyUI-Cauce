"""ComfyUI nodes that wrap native samplers with CAUCE latent operators."""

from __future__ import annotations

from ..cauce.motion import PADDING_MODES
from ..cauce.sigma_transport import (
    SIGMA_EASINGS,
    SIGMA_ENVELOPES,
    SigmaMotionSampler,
    sigma_transport_report_json,
)


SAMPLER_CATEGORY = "CAUCE/H3 Motion"


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
                    {"default": 0.25, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "envelope": (list(SIGMA_ENVELOPES), {"default": "accumulate"}),
                "easing": (list(SIGMA_EASINGS), {"default": "smoothstep"}),
                "padding_mode": (list(PADDING_MODES), {"default": "reflection"}),
            }
        }

    RETURN_TYPES = ("SAMPLER", "STRING")
    RETURN_NAMES = ("sampler", "report_json")
    FUNCTION = "build"
    CATEGORY = SAMPLER_CATEGORY
    DESCRIPTION = (
        "Run deterministic res_multistep with covariant H3 visual-latent "
        "pullbacks on state and solver history; packed audio is unchanged."
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


NODE_CLASS_MAPPINGS = {"CauceSigmaMotionSampler": CauceSigmaMotionSampler}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceSigmaMotionSampler": "CAUCE Sigma-Conditioned H3 Transport"
}
