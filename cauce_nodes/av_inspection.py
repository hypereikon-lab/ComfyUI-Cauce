"""ComfyUI bindings for H3 AV inspection."""

from __future__ import annotations

from ..cauce.av_latent import (
    allocate_av_window_like,
    extract_av_span,
    extract_h3_visual_stream,
    inspect_av_latent,
    plan_av_window,
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


class CauceH3InspectAVLatent:
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

    RETURN_TYPES = ("INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("frame_count", "video_tokens", "audio_tokens", "report_json")
    FUNCTION = "inspect"
    CATEGORY = CATEGORY
    DESCRIPTION = "Validate and report one packed MiniMax H3 AV latent or aligned window."

    def inspect(self, latent, timeline_origin_frame):
        report = inspect_av_latent(
            latent,
            timeline_origin_frame=int(timeline_origin_frame),
        )
        return (
            int(report["frame_count"]),
            int(report["video_tokens"]),
            int(report["audio_tokens"]),
            _json(report),
        )


class CauceH3PlanAVWindow:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "previous_av_latent": ("LATENT",),
                "overlap_frames": (
                    "INT",
                    {"default": 22, "min": 5, "max": 3600, "step": 17},
                ),
                "extension_frames": (
                    "INT",
                    {"default": 119, "min": 17, "max": 3570, "step": 17},
                ),
            }
        }

    RETURN_TYPES = ("CAUCE_H3_AV_LAYOUT", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "layout",
        "window_frames",
        "window_start_frame",
        "overlap_frames",
        "extension_frames",
        "report_json",
    )
    FUNCTION = "plan"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Resolve exact global frame and video/audio-token boundaries for one fresh H3 AV window."
    )

    def plan(self, previous_av_latent, overlap_frames, extension_frames):
        layout = plan_av_window(
            previous_av_latent,
            overlap_frames=int(overlap_frames),
            extension_frames=int(extension_frames),
        )
        return (
            layout,
            int(layout["window_frame_count"]),
            int(layout["window_start_frame"]),
            int(layout["overlap_frames"]),
            int(layout["extension_frames"]),
            _json(layout),
        )


class CauceH3AllocateAVWindow:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "previous_av_latent": ("LATENT",),
                "layout": ("CAUCE_H3_AV_LAYOUT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "allocate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Allocate a zero H3 AV target using a validated absolute timeline layout."

    def allocate(self, previous_av_latent, layout):
        return (
            allocate_av_window_like(
                previous_av_latent,
                layout,
                nested_factory=_nested_factory,
            ),
        )


class CauceH3ExtractAVSpan:
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
                    {"default": 0, "min": 0, "max": 10_000_000},
                ),
                "frame_count": (
                    "INT",
                    {"default": 22, "min": 1, "max": 10_000_000},
                ),
            }
        }

    RETURN_TYPES = ("CAUCE_H3_AV_SPAN", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("span", "frame_count", "video_tokens", "audio_tokens", "report_json")
    FUNCTION = "extract"
    CATEGORY = CATEGORY
    DESCRIPTION = "Extract synchronized H3 video/audio tokens at exact global frame boundaries."

    def extract(self, latent, timeline_origin_frame, start_frame, frame_count):
        span = extract_av_span(
            latent,
            timeline_origin_frame=int(timeline_origin_frame),
            start_frame=int(start_frame),
            frame_count=int(frame_count),
        )
        descriptor = span["descriptor"]
        return (
            span,
            int(descriptor["frame_count"]),
            int(descriptor["video_tokens"]),
            int(descriptor["audio_tokens"]),
            _json(
                {
                    "schema": span["schema"],
                    "descriptor": descriptor,
                    "descriptor_hash": span["descriptor_hash"],
                }
            ),
        )


class CauceH3ExtractVisualStream:
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

    RETURN_TYPES = ("LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("visual_latent", "source_av_latent", "report_json")
    FUNCTION = "extract"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Expose H3 visual state to a visual-only latent tool while retaining the "
        "original AV carrier for an explicit graft."
    )

    def extract(self, latent, timeline_origin_frame):
        visual, report = extract_h3_visual_stream(
            latent,
            timeline_origin_frame=int(timeline_origin_frame),
        )
        return visual, latent, _json(report)


NODE_CLASS_MAPPINGS = {
    "CauceH3InspectAVLatent": CauceH3InspectAVLatent,
    "CauceH3PlanAVWindow": CauceH3PlanAVWindow,
    "CauceH3AllocateAVWindow": CauceH3AllocateAVWindow,
    "CauceH3ExtractAVSpan": CauceH3ExtractAVSpan,
    "CauceH3ExtractVisualStream": CauceH3ExtractVisualStream,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceH3InspectAVLatent": "CAUCE · Inspect H3 AV Latent",
    "CauceH3PlanAVWindow": "CAUCE · Plan H3 AV Window",
    "CauceH3AllocateAVWindow": "CAUCE · Allocate H3 AV Window",
    "CauceH3ExtractAVSpan": "CAUCE · Extract H3 AV Span",
    "CauceH3ExtractVisualStream": "CAUCE · Extract H3 Visual Stream",
}
