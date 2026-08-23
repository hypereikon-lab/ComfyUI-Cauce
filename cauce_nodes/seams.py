"""Native ComfyUI nodes for localized temporal video inpainting."""

from __future__ import annotations

import json

from ..cauce.seams import (
    MASK_CURVES,
    TOKEN_PROJECTIONS,
    add_h3_temporal_inpaint_guides,
    build_seam_window,
    make_seam_window,
    make_seam_plan,
    prepare_h3_temporal_inpaint,
    temporal_inpaint_fields,
    splice_seam_patch,
)


CATEGORY = "CAUCE/Temporal Inpainting"


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
                "repair_seconds_total": (
                    "FLOAT",
                    {"default": 3.0, "min": 1 / 24, "max": 5.0, "step": 1 / 24},
                ),
                "guide_frames": (
                    "INT",
                    {"default": 22, "min": 5, "max": 90, "step": 17},
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
        repair_seconds_total,
        guide_frames,
        maximum_frames,
    ):
        if abs(float(left_fps) - 24.0) > 1e-3 or abs(float(right_fps) - 24.0) > 1e-3:
            raise ValueError("H3 temporal inpainting requires both source videos at 24 fps")
        plan = make_seam_plan(
            int(left_frames.shape[0]),
            int(right_frames.shape[0]),
            context_seconds_per_side=context_seconds_per_side,
            repair_seconds_total=repair_seconds_total,
            guide_frames=guide_frames,
            maximum_frames=maximum_frames,
        )
        working = build_seam_window(left_frames, right_frames, plan)
        window = make_seam_window(plan)
        return working, plan, window, json.dumps(plan, ensure_ascii=False, indent=2)


class CaucePrepareH3TemporalInpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_latent": ("LATENT",),
                "encoded_video_latent": ("LATENT",),
                "seam": ("CAUCE_SEAM",),
                "token_projection": (TOKEN_PROJECTIONS,),
                "sampling_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {
                "generation_support": ("MASK",),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("masked_latent", "mask_report")
    FUNCTION = "prepare"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Inject the encoded source video into H3 and attach the official per-token "
        "temporal denoise mask. Refuses runtimes that cannot preserve masked rows."
    )

    def prepare(
        self,
        target_latent,
        encoded_video_latent,
        seam,
        token_projection="cover",
        sampling_threshold=0.5,
        generation_support=None,
    ):
        latent, report = prepare_h3_temporal_inpaint(
            target_latent,
            encoded_video_latent,
            seam,
            projection=token_projection,
            sampling_threshold=sampling_threshold,
            generation_support=generation_support,
        )
        return latent, json.dumps(report, ensure_ascii=False, indent=2)


class CauceTemporalInpaintFields:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "working_images": ("IMAGE",),
                "seam": ("CAUCE_SEAM",),
                "decoded_blend_frames": (
                    "INT",
                    {"default": 8, "min": 0, "max": 48, "step": 1},
                ),
                "curve": (MASK_CURVES,),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "STRING")
    RETURN_NAMES = (
        "sampling_support",
        "hard_acceptance",
        "output_opacity",
        "field_report",
    )
    FUNCTION = "build"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Compile exact H3 token sampling support, the accepted interval, and a soft "
        "decoded output opacity used only for the final duration-preserving splice."
    )

    def build(
        self,
        working_images,
        seam,
        decoded_blend_frames,
        curve,
    ):
        generation, acceptance, opacity, report = temporal_inpaint_fields(
            working_images,
            seam,
            decoded_blend_frames=decoded_blend_frames,
            curve=curve,
        )
        return generation, acceptance, opacity, json.dumps(
            report, ensure_ascii=False, indent=2
        )


class CauceH3TemporalInpaintGuides:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "target_latent": ("LATENT",),
                "working_images": ("IMAGE",),
                "seam": ("CAUCE_SEAM",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "guide_report")
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Attach two H3 guide clips immediately outside the generated interval so the "
        "model sees incoming and outgoing motion, not only distant endpoint images."
    )

    def apply(self, positive, target_latent, working_images, seam, vae):
        conditioned, report = add_h3_temporal_inpaint_guides(
            positive, target_latent, working_images, seam, vae
        )
        return conditioned, json.dumps(report, ensure_ascii=False, indent=2)


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
                    {"default": 8, "min": 0, "max": 48, "step": 1},
                ),
                "blend_curve": (MASK_CURVES,),
            },
            "optional": {
                "blend_strength": ("MASK",),
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
        decoded_feather_frames=8,
        blend_curve="cosine",
        blend_strength=None,
    ):
        joined, patch, report = splice_seam_patch(
            left_frames,
            right_frames,
            repaired_working_images,
            seam,
            feather_frames=decoded_feather_frames,
            curve=blend_curve,
            blend_strength=blend_strength,
        )
        return joined, patch, json.dumps(report, ensure_ascii=False, indent=2)


NODE_CLASS_MAPPINGS = {
    "CauceBuildSeamWindow": CauceBuildSeamWindow,
    "CauceTemporalInpaintFields": CauceTemporalInpaintFields,
    "CauceH3TemporalInpaintGuides": CauceH3TemporalInpaintGuides,
    "CaucePrepareH3TemporalInpaint": CaucePrepareH3TemporalInpaint,
    "CauceApplySeamPatch": CauceApplySeamPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceBuildSeamWindow": "CAUCE · Build Temporal Inpaint Window",
    "CauceTemporalInpaintFields": "CAUCE · Temporal Inpaint Fields",
    "CauceH3TemporalInpaintGuides": "CAUCE · H3 Temporal Guide Clips",
    "CaucePrepareH3TemporalInpaint": "CAUCE · Prepare H3 Temporal Inpaint",
    "CauceApplySeamPatch": "CAUCE · Splice Temporal Inpaint Patch",
}
