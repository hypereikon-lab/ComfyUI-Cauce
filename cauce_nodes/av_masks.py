"""ComfyUI bindings for H3 AV masks."""

from __future__ import annotations

from ..cauce.av_latent import (
    apply_av_denoise_interval,
    apply_video_denoise_mask,
    clear_av_denoise_mask,
)
from ._shared import (
    existing_h3_keyframes,
    json_report,
    make_nested_tensor,
    require_arbitrary_h3_guides,
)

CATEGORY = "CAUCE/H3 AV Latent"

_json = json_report
_nested_factory = make_nested_tensor
_existing_keyframes = existing_h3_keyframes
_require_native_arbitrary_guides = require_arbitrary_h3_guides


class CauceH3SetAVDenoiseInterval:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "timeline_origin_frame": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10_000_000},
                ),
                "start_frame": (
                    "INT",
                    {"default": 22, "min": 0, "max": 10_000_000},
                ),
                "frame_count": (
                    "INT",
                    {"default": 119, "min": 1, "max": 10_000_000},
                ),
                "inside_strength_video": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "outside_strength_video": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "inside_strength_audio": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "outside_strength_audio": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "fade_in_frames": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10_000_000},
                ),
                "fade_out_frames": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10_000_000},
                ),
                "curve": (["linear", "smoothstep", "smootherstep"],),
                "combine": (["replace", "maximum", "minimum", "multiply"],),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "report_json")
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Attach continuous per-token H3 video/audio noise masks: 1 generates and 0 preserves."
    )

    def apply(
        self,
        latent,
        timeline_origin_frame,
        start_frame,
        frame_count,
        inside_strength_video,
        outside_strength_video,
        inside_strength_audio,
        outside_strength_audio,
        fade_in_frames,
        fade_out_frames,
        curve,
        combine,
    ):
        masked, report = apply_av_denoise_interval(
            latent,
            timeline_origin_frame=int(timeline_origin_frame),
            start_frame=int(start_frame),
            frame_count=int(frame_count),
            inside_strength_video=float(inside_strength_video),
            outside_strength_video=float(outside_strength_video),
            inside_strength_audio=float(inside_strength_audio),
            outside_strength_audio=float(outside_strength_audio),
            fade_in_frames=int(fade_in_frames),
            fade_out_frames=int(fade_out_frames),
            curve=str(curve),
            combine=str(combine),
            nested_factory=_nested_factory,
        )
        return masked, _json(report)


class CauceH3ApplyVideoDenoiseMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "mask": ("MASK",),
                "timeline_origin_frame": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10_000_000},
                ),
                "start_frame": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10_000_000},
                ),
                "frame_count": (
                    "INT",
                    {"default": 124, "min": 1, "max": 10_000_000},
                ),
                "inside_strength_video": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "outside_strength_video": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "audio_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "combine": (["replace", "maximum", "minimum", "multiply"],),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "report_json")
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Project a static or per-frame continuous MASK onto H3 video tokens; "
        "1 generates and 0 preserves."
    )

    def apply(
        self,
        latent,
        mask,
        timeline_origin_frame,
        start_frame,
        frame_count,
        inside_strength_video,
        outside_strength_video,
        audio_strength,
        combine,
    ):
        masked, report = apply_video_denoise_mask(
            latent,
            mask,
            timeline_origin_frame=int(timeline_origin_frame),
            start_frame=int(start_frame),
            frame_count=int(frame_count),
            inside_strength_video=float(inside_strength_video),
            outside_strength_video=float(outside_strength_video),
            audio_strength=float(audio_strength),
            combine=str(combine),
            nested_factory=_nested_factory,
        )
        return masked, _json(report)


class CauceH3ClearAVDenoiseMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "timeline_origin_frame": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10_000_000},
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "BOOLEAN")
    RETURN_NAMES = ("latent", "removed")
    FUNCTION = "clear"
    CATEGORY = CATEGORY
    DESCRIPTION = "Remove a consumed H3 AV noise mask without changing either latent stream."

    def clear(self, latent, timeline_origin_frame):
        return clear_av_denoise_mask(
            latent,
            timeline_origin_frame=int(timeline_origin_frame),
        )


NODE_CLASS_MAPPINGS = {
    "CauceH3SetAVDenoiseInterval": CauceH3SetAVDenoiseInterval,
    "CauceH3ApplyVideoDenoiseMask": CauceH3ApplyVideoDenoiseMask,
    "CauceH3ClearAVDenoiseMask": CauceH3ClearAVDenoiseMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceH3SetAVDenoiseInterval": "CAUCE · Set H3 AV Denoise Interval",
    "CauceH3ApplyVideoDenoiseMask": "CAUCE · Apply H3 Video Denoise Mask",
    "CauceH3ClearAVDenoiseMask": "CAUCE · Clear H3 AV Denoise Mask",
}
