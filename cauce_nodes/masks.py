"""Time-field and native nested H3 mask nodes."""

from __future__ import annotations

import json

from ..cauce.contracts import append_field_span
from ..cauce.masks import combine_nested_masks, compile_nested_av_mask, mask_report


class CauceTimeFieldSpan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "channel": (["video", "audio", "both"], {"default": "both"}),
                "start_seconds": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 99_999.0, "step": 0.001},
                ),
                "end_seconds": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.001, "max": 99_999.0, "step": 0.001},
                ),
                "generate_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {"field": ("CAUCE_FIELD",)},
        }

    RETURN_TYPES = ("CAUCE_FIELD", "STRING")
    RETURN_NAMES = ("field", "field_json")
    FUNCTION = "append"
    CATEGORY = "CAUCE/Masks"
    DESCRIPTION = "Place a technical generate/preserve value on the absolute media clock."

    def append(self, channel, start_seconds, end_seconds, generate_strength, field=None):
        result = append_field_span(
            field,
            channel=channel,
            start=start_seconds,
            end=end_seconds,
            strength=generate_strength,
        )
        return result, json.dumps(result, ensure_ascii=False, indent=2)


class CauceCompileAVMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "window": ("CAUCE_WINDOW",),
                "default_video_generate": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "default_audio_generate": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {"field": ("CAUCE_FIELD",), "spatial_mask": ("MASK",)},
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("masked_latent", "mask_report")
    FUNCTION = "compile"
    CATEGORY = "CAUCE/Masks"
    DESCRIPTION = (
        "Compile rational video/audio fields to H3's nested denoise mask. "
        "Mask polarity is explicit: 1 generates, 0 preserves."
    )

    def compile(
        self,
        latent,
        window,
        default_video_generate,
        default_audio_generate,
        field=None,
        spatial_mask=None,
    ):
        nested = compile_nested_av_mask(
            latent,
            window,
            field=field,
            spatial_mask=spatial_mask,
            default_video=default_video_generate,
            default_audio=default_audio_generate,
        )
        result = dict(latent)
        if result.get("noise_mask") is not None:
            nested = combine_nested_masks(result["noise_mask"], nested)
        result["noise_mask"] = nested
        report = mask_report(
            window,
            field,
            default_video=default_video_generate,
            default_audio=default_audio_generate,
        )
        report["default_video_generate"] = float(default_video_generate)
        report["default_audio_generate"] = float(default_audio_generate)
        report["has_spatial_mask"] = spatial_mask is not None
        return result, json.dumps(report, ensure_ascii=False, indent=2)


NODE_CLASS_MAPPINGS = {
    "CauceTimeFieldSpan": CauceTimeFieldSpan,
    "CauceCompileAVMask": CauceCompileAVMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceTimeFieldSpan": "CAUCE · Time Field Span",
    "CauceCompileAVMask": "CAUCE · Compile H3 AV Mask",
}
