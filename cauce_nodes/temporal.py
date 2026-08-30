"""ComfyUI binding for official H3 guide-retime planning."""

from __future__ import annotations

import json

from ..cauce.temporal import plan_h3_guide_retime


CATEGORY = "CAUCE/Temporal Planning"


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


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
    DESCRIPTION = "Map sparse source frames onto official H3 arbitrary-frame guides."

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


NODE_CLASS_MAPPINGS = {"CaucePlanH3GuideRetime": CaucePlanH3GuideRetime}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CaucePlanH3GuideRetime": "CAUCE · Plan H3 Guide Retime",
}
