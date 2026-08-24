"""ComfyUI bindings for H3 audiovisual-latent persistence."""

from __future__ import annotations

from pathlib import Path

from ..cauce.persistence import (
    load_av_latent,
    resolve_latest_or_indexed,
    safe_output_path,
    save_av_latent_atomic,
)


CATEGORY = "CAUCE/Persistence"


class CauceSaveAVLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "filename_prefix": ("STRING", {"default": "cauce/latents/clip"}),
                "artifact_index": ("INT", {"default": 1, "min": 1, "max": 99999}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("latent_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = "Atomically write one indexed H3 audiovisual latent."

    def save(self, latent, filename_prefix, artifact_index):
        import folder_paths

        root = Path(folder_paths.get_output_directory()).resolve()
        prefix = safe_output_path(root, filename_prefix)
        path = prefix.with_name(f"{prefix.name}_{int(artifact_index):05d}.safetensors")
        return (str(save_av_latent_atomic(latent, path)),)


class CauceLoadAVLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path_or_folder": ("STRING", {"default": "cauce/latents"}),
                "artifact_index": ("INT", {"default": 0, "min": 0, "max": 99999}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "resolved_path")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, path_or_folder, artifact_index):
        try:
            import folder_paths

            path = resolve_latest_or_indexed(
                folder_paths.get_output_directory(),
                path_or_folder,
                artifact_index=int(artifact_index),
            )
            return f"{path}:{path.stat().st_mtime_ns}"
        except Exception:
            return float("NaN")

    def load(self, path_or_folder, artifact_index):
        import folder_paths

        path = resolve_latest_or_indexed(
            folder_paths.get_output_directory(),
            path_or_folder,
            artifact_index=int(artifact_index),
        )
        return load_av_latent(path), str(path)


NODE_CLASS_MAPPINGS = {
    "CauceSaveAVLatent": CauceSaveAVLatent,
    "CauceLoadAVLatent": CauceLoadAVLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceSaveAVLatent": "CAUCE · Save H3 AV Latent",
    "CauceLoadAVLatent": "CAUCE · Load H3 AV Latent",
}
