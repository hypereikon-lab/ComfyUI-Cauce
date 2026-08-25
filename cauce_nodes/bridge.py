"""ComfyUI bindings for native MiniMax H3 guide-bridge preparation."""

from __future__ import annotations

import json

from ..cauce.bridge import (
    VALID_GUIDE_FRAMES,
    VALID_TARGET_FRAMES,
    apply_native_guide_bridge,
    extract_native_guide_bridge_sources,
    plan_native_guide_bridge,
)


CATEGORY = "CAUCE/Native H3"


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


class CauceBuildH3GuideBridge:
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

    RETURN_TYPES = ("IMAGE", "IMAGE", "CAUCE_BRIDGE", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "left_guide",
        "right_guide",
        "bridge",
        "target_frames",
        "right_guide_frame_idx",
        "report_json",
    )
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Select source-tail and source-head clips for two official "
        "MiniMaxH3AddGuide nodes; does not modify H3 conditioning or sampling."
    )

    def build(
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
        plan = plan_native_guide_bridge(
            int(left_images.shape[0]),
            int(right_images.shape[0]),
            guide_frames=int(guide_frames),
            target_frames=int(target_frames),
            fps=left_fps,
        )
        left_guide, right_guide = extract_native_guide_bridge_sources(
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


class CauceApplyH3GuideBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left_images": ("IMAGE",),
                "right_images": ("IMAGE",),
                "generated_target": ("IMAGE",),
                "bridge": ("CAUCE_BRIDGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("joined_images", "generated_bridge", "report_json")
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Discard the two H3 guide intervals and insert only the generated center "
        "between the complete source batches."
    )

    def apply(self, left_images, right_images, generated_target, bridge):
        joined, center, report = apply_native_guide_bridge(
            left_images, right_images, generated_target, bridge
        )
        return joined, center, _json(report)


NODE_CLASS_MAPPINGS = {
    "CauceBuildH3GuideBridge": CauceBuildH3GuideBridge,
    "CauceApplyH3GuideBridge": CauceApplyH3GuideBridge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceBuildH3GuideBridge": "CAUCE · Build H3 Guide Bridge",
    "CauceApplyH3GuideBridge": "CAUCE · Apply H3 Guide Bridge",
}
