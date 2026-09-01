"""ComfyUI bindings for deterministic decoded-media assembly."""

from __future__ import annotations

from ..cauce.assembly import accept_decoded_range, restore_decoded_anchors


CATEGORY = "CAUCE/Assembly"


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


class CauceRestoreDecodedAnchors:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_images": ("IMAGE",),
                "source_images": ("IMAGE",),
                "factor": ([2, 3, 4], {"default": 2}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = (
        "images",
        "delivery_frames",
        "restored_anchor_frames",
        "generated_between_frames",
    )
    FUNCTION = "restore"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Restore exact decoded source frames at every factor-th position in a "
        "densified generated batch; only intervening frames remain generative."
    )

    def restore(self, generated_images, source_images, factor):
        return restore_decoded_anchors(generated_images, source_images, factor)


NODE_CLASS_MAPPINGS = {
    "CauceAcceptDecodedRange": CauceAcceptDecodedRange,
    "CauceRestoreDecodedAnchors": CauceRestoreDecodedAnchors,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceAcceptDecodedRange": "CAUCE · Accept Decoded Range",
    "CauceRestoreDecodedAnchors": "CAUCE · Restore Decoded Anchors",
}
