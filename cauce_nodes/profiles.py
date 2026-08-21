"""Execution-profile and non-mutating preflight nodes."""

from __future__ import annotations

from ..cauce.profiles import PROFILES, format_preflight, get_profile, preflight


class CauceExecutionProfile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "profile_name": (list(PROFILES), {"default": "h3-5090-fl2va-640"})
            }
        }

    RETURN_TYPES = (
        "CAUCE_PROFILE",
        "STRING",
        "STRING",
        "INT",
        "INT",
        "BOOLEAN",
        "INT",
        "INT",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "profile",
        "profile_name",
        "family",
        "width",
        "height",
        "tiled_vae",
        "tile_size",
        "overlap",
        "temporal_size",
        "temporal_overlap",
    )
    FUNCTION = "select"
    CATEGORY = "CAUCE/Runtime"

    def select(self, profile_name):
        profile = get_profile(profile_name)
        return (
            profile,
            profile["name"],
            profile["family"],
            profile["width"],
            profile["height"],
            profile["tiled_vae"],
            profile["vae_tile_size"],
            profile["vae_overlap"],
            profile["vae_temporal_size"],
            profile["vae_temporal_overlap"],
        )


class CaucePreflight:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "profile": ("CAUCE_PROFILE",),
                "minimum_free_reserve_gib": (
                    "FLOAT",
                    {"default": 35.0, "min": 5.0, "max": 1000.0, "step": 1.0},
                ),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING", "CAUCE_PREFLIGHT")
    RETURN_NAMES = ("ready", "report", "preflight")
    FUNCTION = "check"
    CATEGORY = "CAUCE/Runtime"
    DESCRIPTION = "Read-only model, disk, GPU, and runtime checks. Never installs or updates software."

    def check(self, profile, minimum_free_reserve_gib):
        import folder_paths

        report = preflight(
            profile,
            folder_paths.models_dir,
            minimum_free_reserve_gib=minimum_free_reserve_gib,
        )
        return bool(report["ready"]), format_preflight(report), report


NODE_CLASS_MAPPINGS = {
    "CauceExecutionProfile": CauceExecutionProfile,
    "CaucePreflight": CaucePreflight,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceExecutionProfile": "CAUCE · Execution Profile",
    "CaucePreflight": "CAUCE · Preflight",
}
