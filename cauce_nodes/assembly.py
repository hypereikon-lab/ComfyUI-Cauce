"""ComfyUI bindings for deterministic decoded-media assembly."""

from __future__ import annotations

from ..cauce.assembly import accept_decoded_range


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


NODE_CLASS_MAPPINGS = {"CauceAcceptDecodedRange": CauceAcceptDecodedRange}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceAcceptDecodedRange": "CAUCE · Accept Decoded Range",
}
