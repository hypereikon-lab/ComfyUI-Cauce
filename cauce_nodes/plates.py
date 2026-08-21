"""Minimal plate-sketch and Runway handoff nodes."""

from __future__ import annotations

import copy
from pathlib import Path

from ..cauce.contracts import content_hash, make_asset, safe_id
from ..cauce.plates import (
    composite_layer,
    create_canvas,
    domemaster_preview,
    write_plate_sidecars,
)


class CaucePlateCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 32}),
                "background": ("STRING", {"default": "#10190B"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create"
    CATEGORY = "CAUCE/Plates"

    def create(self, width, height, background):
        return (create_canvas(width, height, background),)


class CaucePlateLayer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "canvas": ("IMAGE",),
                "layer": ("IMAGE",),
                "x_percent": (
                    "FLOAT",
                    {"default": 50.0, "min": -100.0, "max": 200.0, "step": 0.1},
                ),
                "y_percent": (
                    "FLOAT",
                    {"default": 50.0, "min": -100.0, "max": 200.0, "step": 0.1},
                ),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01}),
                "rotation": (
                    "FLOAT",
                    {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1},
                ),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blend_mode": (["normal", "screen", "multiply", "add"], {"default": "normal"}),
                "feather_pixels": ("INT", {"default": 24, "min": 0, "max": 1024}),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("composite", "layer_mask")
    FUNCTION = "composite"
    CATEGORY = "CAUCE/Plates"

    def composite(
        self,
        canvas,
        layer,
        x_percent,
        y_percent,
        scale,
        rotation,
        opacity,
        blend_mode,
        feather_pixels,
        mask=None,
    ):
        return composite_layer(
            canvas,
            layer,
            x_percent=x_percent,
            y_percent=y_percent,
            scale=scale,
            rotation=rotation,
            opacity=opacity,
            blend_mode=blend_mode,
            feather_pixels=feather_pixels,
            mask=mask,
        )


class CauceDomemasterPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "outside_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_feather_pixels": ("INT", {"default": 12, "min": 0, "max": 512}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("preview", "dome_mask")
    FUNCTION = "preview"
    CATEGORY = "CAUCE/Plates"

    def preview(self, image, outside_level, edge_feather_pixels):
        return domemaster_preview(image, outside_level, edge_feather_pixels)


class CauceAttachPointImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "point": ("CAUCE_POINT",),
                "image": ("IMAGE",),
                "version_id": ("STRING", {"default": "result_001"}),
                "source": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("CAUCE_POINT", "IMAGE")
    RETURN_NAMES = ("point", "image")
    FUNCTION = "attach"
    CATEGORY = "CAUCE/Plates"

    def attach(self, point, image, version_id, source):
        result = copy.deepcopy(point)
        result.setdefault("assets", []).append(
            make_asset(version_id, "image", source, metadata={"version": version_id})
        )
        result["hash"] = content_hash({key: value for key, value in result.items() if key != "hash"})
        return result, image


class CauceExportPlate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "point": ("CAUCE_POINT",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {"workflow_prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "png_path", "prompt")
    FUNCTION = "export"
    OUTPUT_NODE = True
    CATEGORY = "CAUCE/Plates"
    DESCRIPTION = "Save a plate PNG plus prompt and versioned point manifest for browser-based image generation."

    def export(self, image, point, prompt, workflow_prompt=None, extra_pnginfo=None):
        import folder_paths
        import nodes

        point_id = safe_id(point.get("id", "point"), "point")
        timecode = safe_id(str(point.get("timecode", "00-00-00-00")).replace(":", "-"))
        prefix = f"cauce/plates/{point_id}_{timecode}_plate"
        metadata = dict(extra_pnginfo or {})
        metadata["cauce_point"] = point
        metadata["cauce_prompt"] = prompt
        saved = nodes.SaveImage().save_images(
            image,
            filename_prefix=prefix,
            prompt=workflow_prompt,
            extra_pnginfo=metadata,
        )
        output_root = Path(folder_paths.get_output_directory()).resolve()
        png_path = ""
        for record in saved.get("ui", {}).get("images", []):
            folder = (output_root / record.get("subfolder", "")).resolve()
            if folder != output_root and output_root not in folder.parents:
                raise ValueError("unsafe output path returned by ComfyUI")
            path = folder / record["filename"]
            png_path = str(path)
            write_plate_sidecars(path, point=point, prompt=prompt)
        return {"ui": saved.get("ui", {}), "result": (image, png_path, prompt)}


NODE_CLASS_MAPPINGS = {
    "CaucePlateCanvas": CaucePlateCanvas,
    "CaucePlateLayer": CaucePlateLayer,
    "CauceDomemasterPreview": CauceDomemasterPreview,
    "CauceAttachPointImage": CauceAttachPointImage,
    "CauceExportPlate": CauceExportPlate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CaucePlateCanvas": "CAUCE · Plate Canvas",
    "CaucePlateLayer": "CAUCE · Plate Layer",
    "CauceDomemasterPreview": "CAUCE · Domemaster Preview",
    "CauceAttachPointImage": "CAUCE · Attach Point Image",
    "CauceExportPlate": "CAUCE · Export Plate",
}
