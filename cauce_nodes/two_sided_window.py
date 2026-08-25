"""ComfyUI bindings for an H3 two-sided guide window."""

from __future__ import annotations

import json

from ..cauce.two_sided_window import (
    VALID_GUIDE_FRAMES,
    VALID_TARGET_FRAMES,
    assemble_two_sided_guide_window,
    extract_two_sided_window_guides,
    plan_two_sided_guide_window,
)


CATEGORY = "CAUCE/Native H3"


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


class CaucePrepareH3TwoSidedGuideWindow:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left_images": ("IMAGE",),
                "right_images": ("IMAGE",),
                "left_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0}),
                "right_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0}),
                "guide_frames": (
                    [str(value) for value in VALID_GUIDE_FRAMES],
                    {"default": "22"},
                ),
                "target_frames": (
                    [str(value) for value in VALID_TARGET_FRAMES],
                    {"default": "124"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "CAUCE_H3_GUIDE_WINDOW", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "left_guide",
        "right_guide",
        "window_plan",
        "target_frames",
        "right_guide_frame_idx",
        "report_json",
    )
    FUNCTION = "prepare"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Select a source tail and source head for two official "
        "MiniMaxH3AddGuide nodes; does not modify H3 conditioning or sampling."
    )

    def prepare(
        self,
        left_images,
        right_images,
        left_fps,
        right_fps,
        guide_frames,
        target_frames,
    ):
        if float(left_fps) != float(right_fps):
            raise ValueError("left_fps and right_fps must match")
        plan = plan_two_sided_guide_window(
            int(left_images.shape[0]),
            int(right_images.shape[0]),
            guide_frames=int(guide_frames),
            target_frames=int(target_frames),
            fps=left_fps,
        )
        left_guide, right_guide = extract_two_sided_window_guides(
            left_images, right_images, plan
        )
        return (
            left_guide,
            right_guide,
            plan,
            int(plan["target_frames"]),
            int(plan["right_guide_frame_idx"]),
            _json(plan),
        )


class CauceAssembleH3TwoSidedGuideWindow:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left_images": ("IMAGE",),
                "right_images": ("IMAGE",),
                "generated_target": ("IMAGE",),
                "window_plan": ("CAUCE_H3_GUIDE_WINDOW",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("joined_images", "accepted_generated_range", "report_json")
    FUNCTION = "assemble"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Discard both guide intervals and insert only the accepted generated "
        "range between the complete source batches."
    )

    def assemble(self, left_images, right_images, generated_target, window_plan):
        joined, accepted, report = assemble_two_sided_guide_window(
            left_images, right_images, generated_target, window_plan
        )
        return joined, accepted, _json(report)


NODE_CLASS_MAPPINGS = {
    "CaucePrepareH3TwoSidedGuideWindow": CaucePrepareH3TwoSidedGuideWindow,
    "CauceAssembleH3TwoSidedGuideWindow": CauceAssembleH3TwoSidedGuideWindow,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CaucePrepareH3TwoSidedGuideWindow": "CAUCE · Prepare H3 Two-Sided Guide Window",
    "CauceAssembleH3TwoSidedGuideWindow": "CAUCE · Assemble H3 Two-Sided Guide Window",
}
