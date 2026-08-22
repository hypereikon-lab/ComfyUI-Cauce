"""Native ComfyUI nodes for local temporal seam repair."""

from __future__ import annotations

import json

from ..cauce.seams import (
    build_seam_window,
    make_seam_window,
    make_seam_plan,
    prepare_h3_seam_repair,
    splice_seam_patch,
)


CATEGORY = "CAUCE/Seams"


class CauceBuildSeamWindow:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left_frames": ("IMAGE",),
                "right_frames": ("IMAGE",),
                "left_fps": ("FLOAT", {"default": 24.0}),
                "right_fps": ("FLOAT", {"default": 24.0}),
                "context_seconds_per_side": (
                    "FLOAT",
                    {"default": 2.5, "min": 0.25, "max": 7.5, "step": 1 / 24},
                ),
                "repair_seconds_per_side": (
                    "FLOAT",
                    {"default": 1.0, "min": 1 / 24, "max": 5.0, "step": 1 / 24},
                ),
                "maximum_frames": (
                    "INT",
                    {"default": 362, "min": 124, "max": 362, "step": 17},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "CAUCE_SEAM", "CAUCE_WINDOW", "STRING")
    RETURN_NAMES = ("working_images", "seam", "window", "seam_json")
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Take an opaque tail from A and head from B, place the cut at the center, "
        "validate both sources at 24 fps, and add symmetric guard frames until "
        "the working batch is a legal H3 run."
    )

    def build(
        self,
        left_frames,
        right_frames,
        left_fps,
        right_fps,
        context_seconds_per_side,
        repair_seconds_per_side,
        maximum_frames,
    ):
        if abs(float(left_fps) - 24.0) > 1e-3 or abs(float(right_fps) - 24.0) > 1e-3:
            raise ValueError("Confluence currently requires both source videos at 24 fps")
        plan = make_seam_plan(
            int(left_frames.shape[0]),
            int(right_frames.shape[0]),
            context_seconds_per_side=context_seconds_per_side,
            repair_seconds_per_side=repair_seconds_per_side,
            maximum_frames=maximum_frames,
        )
        working = build_seam_window(left_frames, right_frames, plan)
        window = make_seam_window(plan)
        return working, plan, window, json.dumps(plan, ensure_ascii=False, indent=2)


class CaucePrepareH3SeamRepair:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_latent": ("LATENT",),
                "encoded_video_latent": ("LATENT",),
                "seam": ("CAUCE_SEAM",),
                "latent_feather_frames": (
                    "INT",
                    {"default": 6, "min": 0, "max": 48, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("masked_latent", "mask_report")
    FUNCTION = "prepare"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Inject the encoded 5-second video domain into H3 and denoise only the "
        "central seam. The H3 audio stream stays preserved and silent."
    )

    def prepare(self, target_latent, encoded_video_latent, seam, latent_feather_frames):
        latent, report = prepare_h3_seam_repair(
            target_latent,
            encoded_video_latent,
            seam,
            feather_frames=latent_feather_frames,
        )
        return latent, json.dumps(report, ensure_ascii=False, indent=2)


class CauceApplySeamPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "left_frames": ("IMAGE",),
                "right_frames": ("IMAGE",),
                "repaired_working_images": ("IMAGE",),
                "seam": ("CAUCE_SEAM",),
                "decoded_feather_frames": (
                    "INT",
                    {"default": 6, "min": 0, "max": 48, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("joined_images", "repair_patch", "splice_report")
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Replace only the inner tail/head frames with the generated patch. "
        "Everything outside the repair remains byte-identical and duration is unchanged."
    )

    def apply(
        self,
        left_frames,
        right_frames,
        repaired_working_images,
        seam,
        decoded_feather_frames,
    ):
        joined, patch, report = splice_seam_patch(
            left_frames,
            right_frames,
            repaired_working_images,
            seam,
            feather_frames=decoded_feather_frames,
        )
        return joined, patch, json.dumps(report, ensure_ascii=False, indent=2)


NODE_CLASS_MAPPINGS = {
    "CauceBuildSeamWindow": CauceBuildSeamWindow,
    "CaucePrepareH3SeamRepair": CaucePrepareH3SeamRepair,
    "CauceApplySeamPatch": CauceApplySeamPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceBuildSeamWindow": "CAUCE · Build Confluence Window",
    "CaucePrepareH3SeamRepair": "CAUCE · Prepare H3 Seam Repair",
    "CauceApplySeamPatch": "CAUCE · Apply Confluence Patch",
}
