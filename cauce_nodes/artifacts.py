"""Version receipts and safe H3 AV latent persistence nodes."""

from __future__ import annotations

import json
from pathlib import Path

from ..cauce.artifacts import (
    load_av_latent,
    resolve_latest_or_indexed,
    safe_output_path,
    save_av_latent_atomic,
    write_json_atomic,
)
from ..cauce.contracts import content_hash, make_receipt, safe_id
from ..cauce.profiles import model_manifest


class CauceRunReceipt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "artifact_id": ("STRING", {"default": "clip_001"}),
                "window": ("CAUCE_WINDOW",),
                "profile": ("CAUCE_PROFILE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "sampler": ("STRING", {"default": ""}),
                "scheduler": ("STRING", {"default": ""}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "parents_json": ("STRING", {"default": "[]", "multiline": True}),
            },
            "hidden": {"workflow_prompt": "PROMPT"},
        }

    RETURN_TYPES = ("CAUCE_RECEIPT", "STRING", "STRING")
    RETURN_NAMES = ("receipt", "receipt_json", "receipt_hash")
    FUNCTION = "build"
    CATEGORY = "CAUCE/Artifacts"

    def build(
        self,
        artifact_id,
        window,
        profile,
        seed,
        sampler,
        scheduler,
        steps,
        cfg,
        parents_json,
        workflow_prompt=None,
    ):
        parents = json.loads(parents_json or "[]")
        if not isinstance(parents, list):
            raise ValueError("parents_json must be a JSON list")
        receipt = make_receipt(
            artifact_id,
            parents=parents,
            window=window,
            profile=profile,
            seed=seed,
            sampler=sampler,
            scheduler=scheduler,
            steps=steps,
            cfg=cfg,
            model_hashes={
                item["key"]: item["sha256"] for item in model_manifest(profile)
            },
            workflow_hash=content_hash(workflow_prompt or {}),
        )
        return (
            receipt,
            json.dumps(receipt, ensure_ascii=False, indent=2),
            receipt["receipt_hash"],
        )


class CauceSaveAVLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "filename_prefix": ("STRING", {"default": "cauce/latents/clip"}),
                "artifact_index": ("INT", {"default": 1, "min": 1, "max": 99999}),
            },
            "optional": {"receipt": ("CAUCE_RECEIPT",)},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("latent_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "CAUCE/Artifacts"
    DESCRIPTION = "Atomically overwrite one retry-safe indexed AV latent slot."

    def save(self, latent, filename_prefix, artifact_index, receipt=None):
        import folder_paths

        root = Path(folder_paths.get_output_directory()).resolve()
        prefix = safe_output_path(root, filename_prefix)
        path = prefix.with_name(f"{prefix.name}_{int(artifact_index):05d}.safetensors")
        saved = save_av_latent_atomic(latent, path, receipt=receipt)
        return (str(saved),)


class CauceLoadAVLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path_or_folder": ("STRING", {"default": "cauce/latents"}),
                "artifact_index": ("INT", {"default": 0, "min": 0, "max": 99999}),
            }
        }

    RETURN_TYPES = ("LATENT", "CAUCE_RECEIPT", "STRING")
    RETURN_NAMES = ("latent", "receipt", "resolved_path")
    FUNCTION = "load"
    CATEGORY = "CAUCE/Artifacts"

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
        latent, receipt = load_av_latent(path)
        return latent, receipt or {}, str(path)


class CauceSaveReceipt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "receipt": ("CAUCE_RECEIPT",),
                "relative_path": ("STRING", {"default": "cauce/receipts/clip_001.json"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("receipt_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "CAUCE/Artifacts"

    def save(self, receipt, relative_path):
        import folder_paths

        path = safe_output_path(
            folder_paths.get_output_directory(), relative_path, suffix=".json"
        )
        return (str(write_json_atomic(path, receipt)),)


class CauceCompareReceipts:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"left": ("CAUCE_RECEIPT",), "right": ("CAUCE_RECEIPT",)}}

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("difference_json", "identical")
    FUNCTION = "compare"
    CATEGORY = "CAUCE/Artifacts"

    def compare(self, left, right):
        ignored = {"receipt_hash"}
        keys = sorted((set(left) | set(right)) - ignored)
        differences = {
            key: {"left": left.get(key), "right": right.get(key)}
            for key in keys
            if left.get(key) != right.get(key)
        }
        return json.dumps(differences, ensure_ascii=False, indent=2), not differences


NODE_CLASS_MAPPINGS = {
    "CauceRunReceipt": CauceRunReceipt,
    "CauceSaveAVLatent": CauceSaveAVLatent,
    "CauceLoadAVLatent": CauceLoadAVLatent,
    "CauceSaveReceipt": CauceSaveReceipt,
    "CauceCompareReceipts": CauceCompareReceipts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceRunReceipt": "CAUCE · Run Receipt",
    "CauceSaveAVLatent": "CAUCE · Save AV Latent",
    "CauceLoadAVLatent": "CAUCE · Load AV Latent",
    "CauceSaveReceipt": "CAUCE · Save Receipt",
    "CauceCompareReceipts": "CAUCE · Compare Receipts",
}
