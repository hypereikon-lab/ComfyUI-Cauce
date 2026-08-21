"""H3 latent continuation and decode-domain nodes."""

from __future__ import annotations

from ..cauce.continuity import (
    accept_decoded_window,
    prepare_bridge,
    prepare_continuation,
    resolve_parent_latent,
)


class CaucePrepareContinuation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "target_latent": ("LATENT",),
                "previous_latent": ("LATENT",),
                "context_frames": (
                    ["39", "90", "141", "192", "243", "294", "345"],
                    {"default": "39"},
                ),
                "audio_feather_ticks": (
                    "INT",
                    {"default": 8, "min": 0, "max": 256, "step": 1},
                ),
                "conditioning_mode": (
                    ["mask_only", "mask_plus_guide"],
                    {"default": "mask_only"},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "latent", "trim_frames")
    FUNCTION = "prepare"
    CATEGORY = "CAUCE/Continuity"
    DESCRIPTION = (
        "Copy and preserve a phase-aligned AV parent tail at the target head. "
        "Native guide conditioning is available as an explicit experimental mode."
    )

    def prepare(
        self,
        positive,
        target_latent,
        previous_latent,
        context_frames,
        audio_feather_ticks,
        conditioning_mode,
    ):
        return prepare_continuation(
            positive,
            target_latent,
            previous_latent,
            context_frames=int(context_frames),
            audio_feather_ticks=int(audio_feather_ticks),
            conditioning_mode=conditioning_mode,
        )


class CauceResolveParentLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",), "window": ("CAUCE_WINDOW",)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "resolve"
    CATEGORY = "CAUCE/Continuity"
    DESCRIPTION = (
        "Crop only the unaccepted tail while retaining H3's causal origin. "
        "Use this phase-safe parent for persistence and the next continuation."
    )

    def resolve(self, latent, window):
        return (resolve_parent_latent(latent, window),)


class CaucePrepareBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "target_latent": ("LATENT",),
                "left_parent": ("LATENT",),
                "right_parent": ("LATENT",),
                "context_frames": (
                    ["39", "90", "141", "192", "243", "294", "345"],
                    {"default": "39"},
                ),
                "audio_feather_ticks": (
                    "INT",
                    {"default": 8, "min": 0, "max": 256, "step": 1},
                ),
                "conditioning_mode": (
                    ["mask_only", "mask_plus_guide"],
                    {"default": "mask_only"},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "latent", "middle_frames")
    FUNCTION = "prepare"
    CATEGORY = "CAUCE/Continuity"
    DESCRIPTION = (
        "Protect phase-aligned AV content at both ends of a target and generate "
        "only the missing middle."
    )

    def prepare(
        self,
        positive,
        target_latent,
        left_parent,
        right_parent,
        context_frames,
        audio_feather_ticks,
        conditioning_mode,
    ):
        return prepare_bridge(
            positive,
            target_latent,
            left_parent,
            right_parent,
            context_frames=int(context_frames),
            audio_feather_ticks=int(audio_feather_ticks),
            conditioning_mode=conditioning_mode,
        )


class CauceAcceptDecodedWindow:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "window": ("CAUCE_WINDOW",),
            },
            "optional": {"audio": ("AUDIO",)},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT")
    RETURN_NAMES = ("images", "audio", "accepted_frames")
    FUNCTION = "accept"
    CATEGORY = "CAUCE/Continuity"
    DESCRIPTION = (
        "Accept the window's exact visible-frame range and align audio to the same boundaries."
    )

    def accept(self, images, window, audio=None):
        return accept_decoded_window(images, window, audio)


NODE_CLASS_MAPPINGS = {
    "CaucePrepareContinuation": CaucePrepareContinuation,
    "CauceResolveParentLatent": CauceResolveParentLatent,
    "CaucePrepareBridge": CaucePrepareBridge,
    "CauceAcceptDecodedWindow": CauceAcceptDecodedWindow,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CaucePrepareContinuation": "CAUCE · Prepare H3 Continuation",
    "CauceResolveParentLatent": "CAUCE · Resolve Parent Latent",
    "CaucePrepareBridge": "CAUCE · Prepare H3 AV Bridge",
    "CauceAcceptDecodedWindow": "CAUCE · Accept Decoded Window",
}
