"""ComfyUI bindings for exact temporal planning and H3 mask diagnostics."""

from __future__ import annotations

import json

from ..cauce.temporal import (
    analyze_h3_interleave_projection,
    plan_frame_interpolation,
    plan_h3_guide_retime,
)


CATEGORY = "CAUCE/Temporal Planning"


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


class CaucePlanFrameInterpolation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_frame_count": ("INT", {"default": 124, "min": 2, "max": 100000}),
                "source_fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 1000.0}),
                "multiplier": ("INT", {"default": 2, "min": 1, "max": 32}),
            }
        }

    RETURN_TYPES = (
        "CAUCE_FRAME_INTERPOLATION_PLAN",
        "INT",
        "FLOAT",
        "INT",
        "INT",
        "STRING",
    )
    RETURN_NAMES = (
        "plan",
        "target_frame_count",
        "target_fps",
        "inserted_frame_count",
        "resolved_multiplier",
        "report_json",
    )
    FUNCTION = "plan"
    CATEGORY = CATEGORY
    DESCRIPTION = "Plan exact endpoint-preserving RIFE/FILM-style frame interpolation."

    def plan(self, source_frame_count, source_fps, multiplier):
        plan = plan_frame_interpolation(source_frame_count, multiplier, source_fps)
        return (
            plan,
            int(plan["target_frame_count"]),
            float(plan["target_fps"]["decimal"]),
            int(plan["inserted_frame_count"]),
            int(plan["multiplier"]),
            _json(plan),
        )


class CauceInspectH3InterleaveProjection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_frame_count": ("INT", {"default": 124, "min": 2, "max": 100000}),
                "multiplier": ("INT", {"default": 2, "min": 1, "max": 32}),
            }
        }

    RETURN_TYPES = (
        "CAUCE_H3_INTERLEAVE_REPORT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "BOOLEAN",
        "STRING",
    )
    RETURN_NAMES = (
        "report",
        "resolved_h3_frames",
        "video_tokens",
        "preserve_only_tokens",
        "mixed_tokens",
        "generate_only_tokens",
        "exact_known_frame_preservation",
        "report_json",
    )
    FUNCTION = "inspect"
    CATEGORY = CATEGORY
    DESCRIPTION = "Show how an alternating decoded-frame mask collapses onto real H3 temporal tokens."

    def inspect(self, source_frame_count, multiplier):
        report = analyze_h3_interleave_projection(source_frame_count, multiplier)
        counts = report["token_counts"]
        return (
            report,
            int(report["resolved_h3_frame_count"]),
            int(report["video_token_count"]),
            int(counts["preserve_only"]),
            int(counts["mixed_known_and_missing"]),
            int(counts["generate_only"]),
            bool(report["exact_known_frame_preservation_possible"]),
            _json(report),
        )


class CaucePlanH3GuideRetime:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_frame_count": ("INT", {"default": 124, "min": 2, "max": 100000}),
                "source_fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 1000.0}),
                "duration_scale": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 100.0}),
                "guide_stride_source_frames": (
                    "INT",
                    {"default": 24, "min": 1, "max": 100000},
                ),
            }
        }

    RETURN_TYPES = ("CAUCE_H3_GUIDE_RETIME_PLAN", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "plan",
        "resolved_h3_frames",
        "guide_count",
        "resolved_duration_seconds",
        "report_json",
    )
    FUNCTION = "plan"
    CATEGORY = CATEGORY
    DESCRIPTION = "Map sparse source frames onto official H3 arbitrary-frame guides for creative retiming."

    def plan(self, source_frame_count, source_fps, duration_scale, guide_stride_source_frames):
        plan = plan_h3_guide_retime(
            source_frame_count,
            duration_scale,
            guide_stride_source_frames,
            source_fps,
        )
        return (
            plan,
            int(plan["resolved_h3_frame_count"]),
            int(plan["guide_count"]),
            float(plan["resolved_sample_span_duration"]["decimal"]),
            _json(plan),
        )


NODE_CLASS_MAPPINGS = {
    "CaucePlanFrameInterpolation": CaucePlanFrameInterpolation,
    "CauceInspectH3InterleaveProjection": CauceInspectH3InterleaveProjection,
    "CaucePlanH3GuideRetime": CaucePlanH3GuideRetime,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CaucePlanFrameInterpolation": "CAUCE · Plan Frame Interpolation",
    "CauceInspectH3InterleaveProjection": "CAUCE · Inspect H3 Interleave Projection",
    "CaucePlanH3GuideRetime": "CAUCE · Plan H3 Guide Retime",
}
