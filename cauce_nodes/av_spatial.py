"""ComfyUI bindings for H3 AV spatial."""

from __future__ import annotations

from ..cauce.av_latent import (
    densify_h3_video_tokens,
    expand_av_canvas,
    replace_h3_video_stream,
    resize_h3_av_latent,
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


class CauceH3ExpandAVCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "target_width": (
                    "INT",
                    {"default": 1024, "min": 32, "max": 16384, "step": 32},
                ),
                "target_height": (
                    "INT",
                    {"default": 1024, "min": 32, "max": 16384, "step": 32},
                ),
                "offset_x": (
                    "INT",
                    {"default": 0, "min": 0, "max": 16384, "step": 32},
                ),
                "offset_y": (
                    "INT",
                    {"default": 0, "min": 0, "max": 16384, "step": 32},
                ),
                "source_strength_video": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "new_region_strength_video": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "audio_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "timeline_origin_frame": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10_000_000},
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "report_json")
    FUNCTION = "expand"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Place one packed H3 AV state on a larger 32-pixel-aligned canvas and "
        "mark only the new region for generation."
    )

    def expand(
        self,
        latent,
        target_width,
        target_height,
        offset_x,
        offset_y,
        source_strength_video,
        new_region_strength_video,
        audio_strength,
        timeline_origin_frame,
    ):
        expanded, report = expand_av_canvas(
            latent,
            target_width=int(target_width),
            target_height=int(target_height),
            offset_x=int(offset_x),
            offset_y=int(offset_y),
            source_strength_video=float(source_strength_video),
            new_region_strength_video=float(new_region_strength_video),
            audio_strength=float(audio_strength),
            timeline_origin_frame=int(timeline_origin_frame),
            nested_factory=_nested_factory,
        )
        return expanded, _json(report)


class CauceH3DilateVisualTokens:
    """Create an H3-native slower token lattice for bidirectional infill."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "factor": ([2, 3, 4], {"default": 2}),
                "anchor_denoise": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "gap_denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "feather_tokens": (
                    "INT",
                    {"default": 1, "min": 1, "max": 32},
                ),
                "curve": (["smootherstep", "smoothstep", "linear"],),
                "audio_denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "latent",
        "delivery_frames",
        "h3_target_frames",
        "delivery_fps",
        "trim_tail_frames",
        "report_json",
    )
    FUNCTION = "dilate"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Dilate native H3 visual tokens, retain source tokens as anchors, and mask "
        "the inserted token intervals for bidirectional H3 temporal inpainting."
    )

    def dilate(
        self,
        latent,
        factor,
        anchor_denoise,
        gap_denoise,
        feather_tokens,
        curve,
        audio_denoise,
    ):
        output, report = densify_h3_video_tokens(
            latent,
            factor=int(factor),
            anchor_denoise=float(anchor_denoise),
            gap_denoise=float(gap_denoise),
            feather_tokens=int(feather_tokens),
            curve=str(curve),
            audio_denoise=float(audio_denoise),
            nested_factory=_nested_factory,
        )
        return (
            output,
            int(report["delivery_frame_count"]),
            int(report["h3_target_frame_count"]),
            int(report["delivery_fps"]),
            int(report["decoded_tail_trim_frames"]),
            _json(report),
        )


class CauceH3ResizeAVLatent:
    """Resize H3 visual state without changing its time or audio streams."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "target_width": (
                    "INT",
                    {"default": 1280, "min": 32, "max": 16384, "step": 32},
                ),
                "target_height": (
                    "INT",
                    {"default": 768, "min": 32, "max": 16384, "step": 32},
                ),
                "method": (["bicubic", "bilinear", "nearest-exact", "area"],),
                "video_denoise": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "audio_denoise": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "report_json")
    FUNCTION = "resize"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Resize only native H3 visual state and attach explicit video/audio denoise "
        "masks for a same-model spatial regeneration pass."
    )

    def resize(self, latent, target_width, target_height, method, video_denoise, audio_denoise):
        output, report = resize_h3_av_latent(
            latent,
            target_width=int(target_width),
            target_height=int(target_height),
            method=str(method),
            video_denoise=float(video_denoise),
            audio_denoise=float(audio_denoise),
            nested_factory=_nested_factory,
        )
        return output, _json(report)


class CauceH3ReplaceVisualStream:
    """Replace H3 visual state with a compatible VAE-encoded frame batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_av_latent": ("LATENT",),
                "encoded_video_latent": ("LATENT",),
                "video_denoise": (
                    "FLOAT",
                    {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "audio_denoise": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "report_json")
    FUNCTION = "replace"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Graft a compatible H3-VAE visual latent onto an existing packed AV "
        "carrier for a pixel-upscale, encode, and same-H3 regeneration pass."
    )

    def replace(self, target_av_latent, encoded_video_latent, video_denoise, audio_denoise):
        output, report = replace_h3_video_stream(
            target_av_latent,
            encoded_video_latent,
            video_denoise=float(video_denoise),
            audio_denoise=float(audio_denoise),
            nested_factory=_nested_factory,
        )
        return output, _json(report)


NODE_CLASS_MAPPINGS = {
    "CauceH3ExpandAVCanvas": CauceH3ExpandAVCanvas,
    "CauceH3DilateVisualTokens": CauceH3DilateVisualTokens,
    "CauceH3ResizeAVLatent": CauceH3ResizeAVLatent,
    "CauceH3ReplaceVisualStream": CauceH3ReplaceVisualStream,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceH3ExpandAVCanvas": "CAUCE · Expand H3 AV Canvas",
    "CauceH3DilateVisualTokens": "CAUCE · Dilate H3 Visual Tokens",
    "CauceH3ResizeAVLatent": "CAUCE · Resize H3 AV Latent",
    "CauceH3ReplaceVisualStream": "CAUCE · Replace H3 Visual Stream",
}
