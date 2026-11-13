"""Thin output binding for native MoGe geometry; inference stays upstream."""

from ..cauce.comfy_compat import output_directory
from ..cauce.geometry_persistence import save_geometry_archive


class CauceSaveMoGeGeometry:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "geometry": ("MOGE_GEOMETRY",),
            "filename_prefix": ("STRING", {"default": "cauce/geometry/view-00"}),
            "sample_index_start": ("INT", {"default": 0, "min": 0, "max": 9999999}),
            "audit_stride": ("INT", {"default": 32, "min": 1, "max": 4096}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("archive_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "CAUCE/Persistence"
    DESCRIPTION = "Export native MoGe radial distances, masks and intrinsics without rescaling."

    def save(self, geometry, filename_prefix, sample_index_start, audit_stride):
        root = output_directory()
        path, receipt = save_geometry_archive(geometry, root, filename_prefix,
            sample_index_start=sample_index_start, audit_stride=audit_stride)
        relative = path.relative_to(root.resolve())
        artifact = {"filename": path.name, "subfolder": str(relative.parent).replace("\\", "/"),
                    "type": "output", **receipt}
        return {"ui": {"files": [artifact]}, "result": (str(path),)}


NODE_CLASS_MAPPINGS = {"CauceSaveMoGeGeometry": CauceSaveMoGeGeometry}
NODE_DISPLAY_NAME_MAPPINGS = {"CauceSaveMoGeGeometry": "CAUCE · Save Native MoGe Geometry"}
