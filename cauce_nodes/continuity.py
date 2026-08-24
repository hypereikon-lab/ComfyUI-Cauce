"""ComfyUI bindings for phase-aware H3 continuation."""

from __future__ import annotations

from ..cauce.continuity import (
    VALID_CONTEXT_FRAMES,
    accept_decoded_range,
    prepare_continuation,
    resolve_parent_latent,
)


CATEGORY = "CAUCE/Continuity"


class CaucePrepareContinuation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_latent": ("LATENT",),
                "previous_latent": ("LATENT",),
                "context_frames": (
                    [str(value) for value in VALID_CONTEXT_FRAMES],
                    {"default": "39"},
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "context_frames")
    FUNCTION = "prepare"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Copy a phase-aligned visual tail into the next H3 latent, preserve that "
        "context with a binary mask, and freeze structural audio."
    )

    def prepare(self, target_latent, previous_latent, context_frames):
        return prepare_continuation(
            target_latent,
            previous_latent,
            context_frames=int(context_frames),
        )


class CauceResolveParentLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "accepted_end_frame": (
                    "INT",
                    {"default": 124, "min": 5, "max": 362, "step": 17},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "resolve"
    CATEGORY = CATEGORY
    DESCRIPTION = "Crop a sampled H3 latent at a phase-safe visible-frame endpoint."

    def resolve(self, latent, accepted_end_frame):
        return (resolve_parent_latent(latent, accepted_end_frame),)


class CauceAcceptDecodedRange:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "frame_count": ("INT", {"default": 124, "min": 1, "max": 100000}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "accepted_frames")
    FUNCTION = "accept"
    CATEGORY = CATEGORY
    DESCRIPTION = "Extract an exact visible-frame range from a decoded batch."

    def accept(self, images, start_frame, frame_count):
        return accept_decoded_range(images, start_frame, frame_count)


NODE_CLASS_MAPPINGS = {
    "CaucePrepareContinuation": CaucePrepareContinuation,
    "CauceResolveParentLatent": CauceResolveParentLatent,
    "CauceAcceptDecodedRange": CauceAcceptDecodedRange,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CaucePrepareContinuation": "CAUCE · Prepare H3 Continuation",
    "CauceResolveParentLatent": "CAUCE · Resolve Parent Latent",
    "CauceAcceptDecodedRange": "CAUCE · Accept Decoded Range",
}
