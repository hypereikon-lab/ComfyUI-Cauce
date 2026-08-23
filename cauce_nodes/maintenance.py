"""Comfy-native, root-scoped storage inventory and cleanup nodes."""

from __future__ import annotations

import json
from pathlib import Path

from ..cauce.artifacts import read_json, write_json_atomic
from ..cauce.storage import (
    apply_storage_plan,
    build_storage_plan,
    storage_plan_report,
    validate_storage_plan,
)


CATEGORY = "CAUCE/Maintenance"


def _storage_root(root_kind: str) -> Path:
    import folder_paths

    if root_kind == "input":
        return Path(folder_paths.get_input_directory()).resolve()
    if root_kind == "output":
        return Path(folder_paths.get_output_directory()).resolve()
    raise ValueError("root must be input or output")


def _maintenance_root() -> Path:
    import folder_paths

    return (
        Path(folder_paths.get_user_directory()).resolve()
        / "cauce"
        / "maintenance"
    )


def _plan_path(plan_id: str) -> Path:
    return _maintenance_root() / "plans" / f"{plan_id}.json"


def _receipt_root() -> Path:
    return _maintenance_root() / "receipts"


class CauceStorageInventory:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "root": (["input", "output"], {"default": "output"}),
                "relative_subfolder": ("STRING", {"default": "."}),
                "include_glob": ("STRING", {"default": "*"}),
                "exclude_glob": ("STRING", {"default": ""}),
                "recursive": ("BOOLEAN", {"default": True}),
                "minimum_age_minutes": (
                    "INT",
                    {"default": 0, "min": 0, "max": 5_256_000, "step": 1},
                ),
                "preserve_markers": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CAUCE_STORAGE_PLAN", "STRING", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("plan", "report_json", "confirmation", "file_count", "total_gib")
    FUNCTION = "inventory"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Read-only recursive inventory of ComfyUI input or output. Produces a hashed "
        "plan for CAUCE Storage Cleanup; it never scans models or custom_nodes."
    )

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("NaN")

    def inventory(
        self,
        root,
        relative_subfolder,
        include_glob,
        exclude_glob,
        recursive,
        minimum_age_minutes,
        preserve_markers,
    ):
        plan = build_storage_plan(
            _storage_root(root),
            root_kind=root,
            relative_subfolder=relative_subfolder,
            include_glob=include_glob,
            exclude_glob=exclude_glob,
            recursive=recursive,
            minimum_age_seconds=int(minimum_age_minutes) * 60,
            preserve_markers=preserve_markers,
        )
        report = json.dumps(storage_plan_report(plan), ensure_ascii=False, indent=2)
        summary = plan["summary"]
        return {
            "ui": {"text": [report]},
            "result": (
                plan,
                report,
                plan["confirmation"],
                summary["file_count"],
                float(summary["total_gib"]),
            ),
        }


class CauceStorageCleanup:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("CAUCE_STORAGE_PLAN",),
                "armed": ("BOOLEAN", {"default": False}),
                "confirmation": ("STRING", {"default": ""}),
                "remove_empty_directories": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("receipt_json", "deleted_count", "freed_gib", "receipt_path")
    FUNCTION = "cleanup"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Stages a plan while armed=false. A later armed=true run deletes only unchanged "
        "files from that exact staged plan and requires its confirmation code."
    )

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("NaN")

    def cleanup(self, plan, armed, confirmation, remove_empty_directories):
        validate_storage_plan(plan)
        staged_path = _plan_path(plan["plan_id"])
        if not armed:
            write_json_atomic(staged_path, plan)
        else:
            if not staged_path.is_file():
                raise ValueError(
                    "this storage plan has not been staged; run once with armed=false"
                )
            staged = read_json(staged_path)
            validate_storage_plan(staged)
            if (
                staged["plan_id"] != plan["plan_id"]
                or staged["root_kind"] != plan["root_kind"]
                or staged["scope"] != plan["scope"]
                or staged["entries"] != plan["entries"]
            ):
                raise ValueError("the staged storage plan does not match the live plan")
        receipt = apply_storage_plan(
            _storage_root(plan["root_kind"]),
            plan,
            armed=armed,
            confirmation=confirmation,
            remove_empty_directories=remove_empty_directories,
        )
        receipt_path = ""
        if not armed:
            receipt["status"] = "staged"
            receipt["staged_plan_path"] = str(staged_path)
        if armed:
            target = _receipt_root() / (
                f"{receipt['completed_at_ns']}_{receipt['root_kind']}_"
                f"{receipt['plan_id'][:12]}.json"
            )
            receipt_path = str(write_json_atomic(target, receipt))
        encoded = json.dumps(receipt, ensure_ascii=False, indent=2)
        return {
            "ui": {"text": [encoded]},
            "result": (
                encoded,
                receipt["deleted_count"],
                receipt["freed_bytes"] / 1024**3,
                receipt_path,
            ),
        }


NODE_CLASS_MAPPINGS = {
    "CauceStorageInventory": CauceStorageInventory,
    "CauceStorageCleanup": CauceStorageCleanup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceStorageInventory": "CAUCE · Storage Inventory",
    "CauceStorageCleanup": "CAUCE · Storage Cleanup",
}
