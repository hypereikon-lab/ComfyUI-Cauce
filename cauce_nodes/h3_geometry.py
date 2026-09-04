"""ComfyUI binding for CAUCE's experimental H3 geometry-coordinate patch."""

from __future__ import annotations

from ..cauce.h3_geometry import (
    DOMEMASTER_PROFILE,
    DOMEMASTER_WRAPPER_KEY,
    H3DomemasterCoordinatePatch,
)
from ._shared import json_report


CATEGORY = "CAUCE/H3 Model"


class CauceH3DomemasterCoordinates:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "strength": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "include_keyframes": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "outside_disc": (["stock", "rim"],),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report_json")
    FUNCTION = "patch"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Experimental MiniMax H3 RoPE coordinate warp for an equidistant 180-degree "
        "domemaster. It changes token coordinates only; it is not a trained lens adapter."
    )

    def patch(self, model, strength, include_keyframes, outside_disc):
        amount = float(strength)
        include_keyframes = bool(include_keyframes)
        outside_disc = str(outside_disc)

        # A zero-strength control must be structurally identical to stock H3.
        # Even a mathematically inert diffusion-model wrapper can change graph/model
        # patch identity and therefore execution or cache behavior in ComfyUI.
        if amount == 0.0:
            report = {
                "schema": "cauce.h3-domemaster-coordinate-node/1",
                "profile": DOMEMASTER_PROFILE,
                "strength": amount,
                "include_keyframes": include_keyframes,
                "outside_disc": outside_disc,
                "scope": [],
                "mutates": [],
                "does_not_mutate": [
                    "model object",
                    "model wrappers",
                    "pixels",
                    "latents",
                    "weights",
                    "prompts",
                    "text rows",
                    "audio rows",
                    "Ref2VA reference rows",
                    "sampling schedule",
                ],
                "bypassed_bit_exactly": True,
                "evidence": "structural-identity-control",
            }
            return model, json_report(report)

        import comfy.patcher_extension

        patched = model.clone()
        coordinate_patch = H3DomemasterCoordinatePatch(
            strength=amount,
            include_keyframes=include_keyframes,
            outside_disc=outside_disc,
        )
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            DOMEMASTER_WRAPPER_KEY,
            coordinate_patch,
        )
        report = {
            "schema": "cauce.h3-domemaster-coordinate-node/1",
            "profile": DOMEMASTER_PROFILE,
            "strength": amount,
            "include_keyframes": include_keyframes,
            "outside_disc": outside_disc,
            "scope": ["video", "cond"] if include_keyframes else ["video"],
            "mutates": ["packed_layout.position_ids[h,w]"],
            "does_not_mutate": [
                "pixels",
                "latents",
                "weights",
                "prompts",
                "text rows",
                "audio rows",
                "Ref2VA reference rows",
                "sampling schedule",
            ],
            "bypassed_bit_exactly": False,
            "evidence": "experimental-inference-ablation",
        }
        return patched, json_report(report)


NODE_CLASS_MAPPINGS = {
    "CauceH3DomemasterCoordinates": CauceH3DomemasterCoordinates,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceH3DomemasterCoordinates": "CAUCE · H3 Domemaster Coordinates",
}
