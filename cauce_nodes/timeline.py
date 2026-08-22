"""ComfyUI nodes for CAUCE media-first timeline data."""

from __future__ import annotations

import json

from ..cauce.contracts import (
    append_timeline_item,
    make_point,
    make_project,
    make_decode_domain,
    make_span,
    make_timeline,
    make_window,
    window_summary,
)


CATEGORY = "CAUCE/Timeline"


class CauceProject:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_id": ("STRING", {"default": "inside-valdivia"}),
                "title": ("STRING", {"default": "Inside Valdivia"}),
                "output_root": ("STRING", {"default": "cauce"}),
            },
            "optional": {"timeline": ("CAUCE_TIMELINE",)},
        }

    RETURN_TYPES = ("CAUCE_PROJECT", "STRING")
    RETURN_NAMES = ("project", "project_json")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, project_id, title, output_root, timeline=None):
        project = make_project(
            project_id, title, timeline=timeline, output_root=output_root
        )
        return project, json.dumps(project, ensure_ascii=False, indent=2)


class CauceTimelinePoint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "point_id": ("STRING", {"default": "point_001"}),
                "master_seconds": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 99_999.0, "step": 0.001},
                ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("CAUCE_POINT", "CAUCE_ITEM", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("point", "timeline_item", "prompt", "master_seconds", "timecode")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "An opaque image/prompt position on the absolute CAUCE clock."

    def build(self, point_id, master_seconds, prompt):
        point = make_point(point_id, master_seconds, prompt)
        return point, point, prompt, float(master_seconds), point["timecode"]


class CauceMediaSpan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "span_id": ("STRING", {"default": "span_001"}),
                "kind": (["image", "video", "audio", "mask", "latent"],),
                "start_seconds": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 99_999.0, "step": 0.001},
                ),
                "end_seconds": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.001, "max": 99_999.0, "step": 0.001},
                ),
                "source": ("STRING", {"default": ""}),
                "offset_seconds": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 99_999.0, "step": 0.001},
                ),
            }
        }

    RETURN_TYPES = ("CAUCE_SPAN", "CAUCE_ITEM", "STRING")
    RETURN_NAMES = ("span", "timeline_item", "span_json")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = "Place opaque image, video, audio, mask, or latent media on the master clock."

    def build(self, span_id, kind, start_seconds, end_seconds, source, offset_seconds):
        span = make_span(
            span_id,
            kind,
            start_seconds,
            end_seconds,
            source=source,
            offset=offset_seconds,
        )
        return span, span, json.dumps(span, ensure_ascii=False, indent=2)


class CauceGenerationWindow:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "window_id": ("STRING", {"default": "window_001"}),
                "accepted_start_seconds": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 99_999.0, "step": 0.001},
                ),
                "accepted_duration_seconds": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.1, "max": 15.0, "step": 0.001},
                ),
                "context_frames": (
                    [str(value) for value in (0, *range(5, 346, 17))],
                    {"default": "0"},
                ),
                "duplicate_prefix_frames": (["0", "5"], {"default": "0"}),
                "snap_mode": (["ceil", "nearest", "floor"], {"default": "ceil"}),
                "accept_mode": (
                    ["nearest_run", "floor_run", "ceil_run", "exact_frames", "full_render"],
                    {"default": "nearest_run"},
                ),
                "maximum_frames": (
                    "INT",
                    {"default": 362, "min": 124, "max": 3600, "step": 17},
                ),
            }
        }

    RETURN_TYPES = (
        "CAUCE_WINDOW",
        "CAUCE_ITEM",
        "INT",
        "FLOAT",
        "INT",
        "FLOAT",
        "FLOAT",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "window",
        "timeline_item",
        "length",
        "fps",
        "accepted_offset_frames",
        "accepted_duration_seconds",
        "accepted_end_seconds",
        "summary",
        "window_json",
    )
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Compile context, duplicate-prefix, render, accepted, and discarded ranges "
        "onto H3's exact 17k+5 frame grid."
    )

    def build(
        self,
        window_id,
        accepted_start_seconds,
        accepted_duration_seconds,
        context_frames,
        duplicate_prefix_frames,
        snap_mode,
        accept_mode,
        maximum_frames,
    ):
        window = make_window(
            window_id,
            accepted_start_seconds,
            accepted_duration_seconds,
            context_frames=int(context_frames),
            duplicate_prefix_frames=int(duplicate_prefix_frames),
            snap_mode=snap_mode,
            accept_mode=accept_mode,
            maximum_frames=int(maximum_frames),
        )
        return (
            window,
            window,
            int(window["shape"]["pixel_frames"]),
            24.0,
            int(window["accepted_offset_frames"]),
            float(window["accepted_range"]["duration_seconds"]),
            float(window["accepted_range"]["end_seconds"]),
            window_summary(window),
            json.dumps(window, ensure_ascii=False, indent=2),
        )


class CauceEmptyTimeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"timeline_id": ("STRING", {"default": "main"})}}

    RETURN_TYPES = ("CAUCE_TIMELINE", "STRING")
    RETURN_NAMES = ("timeline", "timeline_json")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, timeline_id):
        timeline = make_timeline(timeline_id)
        return timeline, json.dumps(timeline, ensure_ascii=False, indent=2)


class CauceAppendTimeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"item": ("CAUCE_ITEM",)},
            "optional": {"timeline": ("CAUCE_TIMELINE",)},
        }

    RETURN_TYPES = ("CAUCE_TIMELINE", "STRING")
    RETURN_NAMES = ("timeline", "timeline_json")
    FUNCTION = "append"
    CATEGORY = CATEGORY

    def append(self, item, timeline=None):
        result = append_timeline_item(timeline, item)
        return result, json.dumps(result, ensure_ascii=False, indent=2)


class CauceDecodeDomain:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "domain_id": ("STRING", {"default": "decode_001"}),
                "start_seconds": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 99_999.0, "step": 0.001},
                ),
                "end_seconds": (
                    "FLOAT",
                    {"default": 30.0, "min": 0.001, "max": 99_999.0, "step": 0.001},
                ),
                "artifact_ids_json": ("STRING", {"default": "[]", "multiline": True}),
            }
        }

    RETURN_TYPES = ("CAUCE_DECODE_DOMAIN", "STRING")
    RETURN_NAMES = ("decode_domain", "domain_json")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, domain_id, start_seconds, end_seconds, artifact_ids_json):
        artifact_ids = json.loads(artifact_ids_json or "[]")
        if not isinstance(artifact_ids, list):
            raise ValueError("artifact_ids_json must be a JSON list")
        domain = make_decode_domain(
            domain_id,
            start_seconds,
            end_seconds,
            artifact_ids=[str(value) for value in artifact_ids],
        )
        return domain, json.dumps(domain, ensure_ascii=False, indent=2)


NODE_CLASS_MAPPINGS = {
    "CauceProject": CauceProject,
    "CauceTimelinePoint": CauceTimelinePoint,
    "CauceMediaSpan": CauceMediaSpan,
    "CauceGenerationWindow": CauceGenerationWindow,
    "CauceEmptyTimeline": CauceEmptyTimeline,
    "CauceAppendTimeline": CauceAppendTimeline,
    "CauceDecodeDomain": CauceDecodeDomain,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceProject": "CAUCE · Project",
    "CauceTimelinePoint": "CAUCE · Timeline Point",
    "CauceMediaSpan": "CAUCE · Media Span",
    "CauceGenerationWindow": "CAUCE · Compile Window",
    "CauceEmptyTimeline": "CAUCE · Empty Timeline",
    "CauceAppendTimeline": "CAUCE · Append Timeline Item",
    "CauceDecodeDomain": "CAUCE · Decode Domain",
}
