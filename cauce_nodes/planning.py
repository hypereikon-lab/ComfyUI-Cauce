"""ComfyUI bindings for visible H3 preprocessing and conditioning inspection."""

from __future__ import annotations

import json

from ..cauce.conditioning import inspect_h3_conditioning
from ..cauce.h3_inputs import (
    prepare_h3_guide_clip,
    prepare_h3_reference_clip,
    resolve_h3_target_shape,
)


CATEGORY = "CAUCE/H3 Planning"


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


class CauceH3ResolveTargetShape:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "requested_frames": (
                    "INT",
                    {"default": 124, "min": 1, "max": 100000},
                ),
                "width": (
                    "INT",
                    {"default": 1344, "min": 32, "max": 16384, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 32, "max": 16384, "step": 32},
                ),
            }
        }

    RETURN_TYPES = (
        "CAUCE_H3_TARGET_PLAN",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "FLOAT",
        "BOOLEAN",
        "STRING",
    )
    RETURN_NAMES = (
        "plan",
        "resolved_frames",
        "width",
        "height",
        "video_tokens",
        "audio_tokens",
        "duration_seconds",
        "inside_trained_range",
        "report_json",
    )
    FUNCTION = "resolve"
    CATEGORY = CATEGORY
    DESCRIPTION = "Expose H3 target lattice, duration, token counts, and trained-range status."

    def resolve(self, requested_frames, width, height):
        plan = resolve_h3_target_shape(requested_frames, width, height)
        return (
            plan,
            int(plan["resolved_frames"]),
            int(plan["width"]),
            int(plan["height"]),
            int(plan["video_tokens"]),
            int(plan["audio_tokens"]),
            float(plan["duration_seconds"]),
            bool(plan["inside_trained_range"]),
            _json(plan),
        )


class CauceH3PrepareGuideClip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "target_av_latent": ("LATENT",),
                "timeline_origin_frame": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10_000_000},
                ),
                "frame_idx": (
                    "INT",
                    {"default": 0, "min": -9999, "max": 9999},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "CAUCE_H3_GUIDE_PLAN", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "images",
        "plan",
        "accepted_frames",
        "discarded_tail_frames",
        "resolved_frame_idx",
        "guide_end_frame",
        "report_json",
    )
    FUNCTION = "prepare"
    CATEGORY = CATEGORY
    DESCRIPTION = "Make official H3 AddGuide clip truncation and placement explicit."

    def prepare(self, images, target_av_latent, timeline_origin_frame, frame_idx):
        accepted, plan = prepare_h3_guide_clip(
            images,
            target_av_latent,
            frame_idx,
            timeline_origin_frame=int(timeline_origin_frame),
        )
        return (
            accepted,
            plan,
            int(plan["accepted_frames"]),
            int(plan["discarded_tail_frames"]),
            int(plan["resolved_frame_idx"]),
            int(plan["guide_range"][1]),
            _json(plan),
        )


class CauceH3PrepareReferenceClip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "requested_target_frames": (
                    "INT",
                    {"default": 124, "min": 5, "max": 3600, "step": 17},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "CAUCE_H3_REFERENCE_PLAN", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "images",
        "plan",
        "accepted_frames",
        "discarded_tail_frames",
        "qwen_sample_count",
        "report_json",
    )
    FUNCTION = "prepare"
    CATEGORY = CATEGORY
    DESCRIPTION = "Expose official Ref2VA target clamp, lattice trim, and 2 fps Qwen samples."

    def prepare(self, images, requested_target_frames):
        accepted, plan = prepare_h3_reference_clip(images, requested_target_frames)
        return (
            accepted,
            plan,
            int(plan["accepted_frames"]),
            int(plan["discarded_tail_frames"]),
            int(plan["qwen_sample_count"]),
            _json(plan),
        )


class CauceH3InspectConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "target_av_latent": ("LATENT",),
                "timeline_origin_frame": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10_000_000},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "positive",
        "target_frames",
        "keyframe_count",
        "reference_count",
        "report_json",
    )
    FUNCTION = "inspect"
    CATEGORY = CATEGORY
    DESCRIPTION = "Read and validate active H3 keyframe/reference conditioning without mutation."

    def inspect(self, positive, target_av_latent, timeline_origin_frame):
        report = inspect_h3_conditioning(
            positive,
            target_av_latent,
            timeline_origin_frame=int(timeline_origin_frame),
        )
        return (
            positive,
            int(report["target_frames"]),
            int(report["keyframe_count"]),
            int(report["reference_count"]),
            _json(report),
        )


NODE_CLASS_MAPPINGS = {
    "CauceH3ResolveTargetShape": CauceH3ResolveTargetShape,
    "CauceH3PrepareGuideClip": CauceH3PrepareGuideClip,
    "CauceH3PrepareReferenceClip": CauceH3PrepareReferenceClip,
    "CauceH3InspectConditioning": CauceH3InspectConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceH3ResolveTargetShape": "CAUCE · Resolve H3 Target Shape",
    "CauceH3PrepareGuideClip": "CAUCE · Prepare H3 Guide Clip",
    "CauceH3PrepareReferenceClip": "CAUCE · Prepare H3 Reference Clip",
    "CauceH3InspectConditioning": "CAUCE · Inspect H3 Conditioning",
}
