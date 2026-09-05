"""Thin binding for the fixed Zenith-180 RoPE experiment."""
from ..cauce.h3_zenith_rope import H3ZenithRoPEPatch, PROFILE, WRAPPER_KEY, validate_options
from ._shared import json_report


class CauceH3ZenithRoPE:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "strength": ("FLOAT", {"default": 0.0, "min": 0., "max": 1., "step": .05}),
            "low_frequency_count": ("INT", {"default": 8, "min": 1, "max": 15}),
            "include_keyframes": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report_json")
    FUNCTION = "patch"
    CATEGORY = "CAUCE/H3 Model"
    DESCRIPTION = "Experimental Zenith-180 ray phases on selected spatial H3 RoPE bands. Square, unremapped domemaster only. No geometry guarantee."

    def patch(self, model, strength, low_frequency_count, include_keyframes):
        validate_options(strength, low_frequency_count)
        report = {"profile": PROFILE, "strength": strength,
                  "low_frequency_count": low_frequency_count,
                  "include_keyframes": include_keyframes,
                  "structural_bypass": strength == 0,
                  "evidence": "experimental-inference-ablation"}
        if strength == 0:
            return model, json_report(report)
        import comfy.patcher_extension
        patched = model.clone()
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY, H3ZenithRoPEPatch(strength, low_frequency_count, include_keyframes),
        )
        return patched, json_report(report)


NODE_CLASS_MAPPINGS = {"CauceH3ZenithRoPE": CauceH3ZenithRoPE}
NODE_DISPLAY_NAME_MAPPINGS = {"CauceH3ZenithRoPE": "CAUCE · H3 Zenith RoPE (Experimental)"}
