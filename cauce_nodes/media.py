"""Minimal nodes for opaque image batches."""

from __future__ import annotations

from ..cauce.media import select_image_frame


class CauceSelectImageFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_index": (
                    "INT",
                    {"default": -1, "min": -999_999, "max": 999_999, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "resolved_frame_index")
    FUNCTION = "select"
    CATEGORY = "CAUCE/Media"
    DESCRIPTION = (
        "Select one opaque image from a batch. Negative indices count from the end; "
        "-1 is the final frame."
    )

    def select(self, images, frame_index):
        return select_image_frame(images, frame_index)


NODE_CLASS_MAPPINGS = {"CauceSelectImageFrame": CauceSelectImageFrame}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceSelectImageFrame": "CAUCE · Select Image Frame"
}
