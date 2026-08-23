"""Scoped, two-phase maintenance for ComfyUI input and output storage."""

from __future__ import annotations

import fnmatch
import hashlib
import os
from pathlib import Path
import time
from typing import Any, Iterable

from .contracts import canonical_json


STORAGE_PLAN_FORMAT = "cauce.storage-plan/1"
STORAGE_RECEIPT_FORMAT = "cauce.storage-receipt/1"
STORAGE_ROOT_KINDS = {"input", "output"}
PROTECTED_MARKER_NAMES = {
    ".gitkeep",
    ".keep",
    "_input_images_will_be_put_here",
    "_output_images_will_be_put_here",
}


def _patterns(value: str, *, default: str = "") -> tuple[str, ...]:
    items = tuple(
        item.strip()
        for line in str(value or "").splitlines()
        for item in line.replace(";", ",").split(",")
        if item.strip()
    )
    return items or ((default,) if default else ())


def _is_within(root: Path, target: Path) -> bool:
    return target == root or root in target.parents


def _relative_parts(relative: str) -> tuple[str, ...]:
    raw = Path(str(relative or ".").strip() or ".")
    if raw.is_absolute():
        raise ValueError("storage paths must be relative to the selected root")
    parts = tuple(part for part in raw.parts if part not in {"", "."})
    if ".." in parts:
        raise ValueError("storage paths cannot contain '..'")
    return parts


def _reject_symlink_components(root: Path, parts: Iterable[str]) -> None:
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"storage path contains a symlink: {current.name}")


def resolve_scoped_directory(root: str | Path, relative: str = ".") -> Path:
    """Resolve one directory without allowing absolute paths, escapes, or symlinks."""

    base = Path(root).expanduser().resolve(strict=True)
    if not base.is_dir():
        raise NotADirectoryError(base)
    parts = _relative_parts(relative)
    _reject_symlink_components(base, parts)
    target = base.joinpath(*parts).resolve(strict=False)
    if not _is_within(base, target):
        raise ValueError("storage path escapes the selected root")
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(target)
    return target


def resolve_planned_file(root: str | Path, relative_path: str) -> Path:
    """Resolve one planned file under a root and reject symlink components."""

    base = Path(root).expanduser().resolve(strict=True)
    parts = _relative_parts(relative_path)
    if not parts:
        raise ValueError("a planned file path cannot be empty")
    _reject_symlink_components(base, parts)
    target = base.joinpath(*parts).resolve(strict=False)
    if not _is_within(base, target) or target == base:
        raise ValueError("planned file escapes the selected root")
    return target


def _walk_files(selected: Path, *, recursive: bool) -> tuple[list[Path], list[dict[str, str]]]:
    files: list[Path] = []
    skipped: list[dict[str, str]] = []
    if not selected.exists():
        return files, skipped

    if not recursive:
        for item in sorted(selected.iterdir(), key=lambda path: path.name.casefold()):
            if item.is_symlink():
                skipped.append({"path": item.name, "reason": "symlink"})
            elif item.is_file():
                files.append(item)
        return files, skipped

    for directory, dirnames, filenames in os.walk(selected, followlinks=False):
        directory_path = Path(directory)
        safe_directories: list[str] = []
        for dirname in sorted(dirnames, key=str.casefold):
            candidate = directory_path / dirname
            if candidate.is_symlink():
                skipped.append({"path": str(candidate), "reason": "symlink_directory"})
            else:
                safe_directories.append(dirname)
        dirnames[:] = safe_directories
        for filename in sorted(filenames, key=str.casefold):
            candidate = directory_path / filename
            if candidate.is_symlink():
                skipped.append({"path": str(candidate), "reason": "symlink"})
            elif candidate.is_file():
                files.append(candidate)
    return files, skipped


def _matches(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _plan_payload(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": plan.get("format"),
        "root_kind": plan.get("root_kind"),
        "scope": plan.get("scope"),
        "entries": plan.get("entries"),
    }


def _plan_id(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def validate_storage_plan(plan: dict[str, Any]) -> str:
    if not isinstance(plan, dict) or plan.get("format") != STORAGE_PLAN_FORMAT:
        raise ValueError("unrecognized CAUCE storage plan")
    if plan.get("root_kind") not in STORAGE_ROOT_KINDS:
        raise ValueError("storage plan root must be input or output")
    if not isinstance(plan.get("scope"), dict) or not isinstance(plan.get("entries"), list):
        raise ValueError("malformed CAUCE storage plan")
    for entry in plan["entries"]:
        if not isinstance(entry, dict):
            raise ValueError("malformed storage plan entry")
        if not isinstance(entry.get("relative_path"), str):
            raise ValueError("storage plan entry is missing its relative path")
        if not isinstance(entry.get("size"), int) or entry["size"] < 0:
            raise ValueError("storage plan entry has an invalid size")
        if not isinstance(entry.get("mtime_ns"), int) or entry["mtime_ns"] < 0:
            raise ValueError("storage plan entry has an invalid modification time")
    expected = _plan_id(_plan_payload(plan))
    if plan.get("plan_id") != expected:
        raise ValueError("storage plan hash does not match its contents")
    expected_confirmation = f"DELETE {plan['root_kind'].upper()} {expected[:12].upper()}"
    if plan.get("confirmation") != expected_confirmation:
        raise ValueError("storage plan confirmation does not match its contents")
    return expected


def build_storage_plan(
    root: str | Path,
    *,
    root_kind: str,
    relative_subfolder: str = ".",
    include_glob: str = "*",
    exclude_glob: str = "",
    recursive: bool = True,
    minimum_age_seconds: int = 0,
    preserve_markers: bool = True,
) -> dict[str, Any]:
    """Inventory physical files and return a stable deletion plan."""

    root_kind = str(root_kind).strip().lower()
    if root_kind not in STORAGE_ROOT_KINDS:
        raise ValueError("root_kind must be input or output")
    if int(minimum_age_seconds) < 0:
        raise ValueError("minimum_age_seconds cannot be negative")

    base = Path(root).expanduser().resolve(strict=True)
    selected = resolve_scoped_directory(base, relative_subfolder)
    includes = _patterns(include_glob, default="*")
    excludes = _patterns(exclude_glob)
    candidates, traversal_skips = _walk_files(selected, recursive=bool(recursive))
    now_ns = time.time_ns()
    minimum_age_ns = int(minimum_age_seconds) * 1_000_000_000
    entries: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for raw_skip in traversal_skips:
        raw_path = Path(raw_skip["path"])
        try:
            relative = raw_path.relative_to(base).as_posix()
        except ValueError:
            relative = raw_path.name
        skipped.append({"relative_path": relative, "reason": raw_skip["reason"]})

    for path in candidates:
        relative = path.relative_to(base).as_posix()
        if preserve_markers and path.name in PROTECTED_MARKER_NAMES:
            skipped.append({"relative_path": relative, "reason": "protected_marker"})
            continue
        if not _matches(relative, includes):
            continue
        if excludes and _matches(relative, excludes):
            skipped.append({"relative_path": relative, "reason": "excluded_glob"})
            continue
        stat = path.stat()
        if minimum_age_ns and now_ns - stat.st_mtime_ns < minimum_age_ns:
            skipped.append({"relative_path": relative, "reason": "too_recent"})
            continue
        entries.append(
            {
                "relative_path": relative,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )

    entries.sort(key=lambda item: item["relative_path"].casefold())
    skipped.sort(key=lambda item: (item["relative_path"].casefold(), item["reason"]))
    relative_selected = selected.relative_to(base).as_posix()
    payload = {
        "format": STORAGE_PLAN_FORMAT,
        "root_kind": root_kind,
        "scope": {
            "relative_subfolder": "." if relative_selected == "." else relative_selected,
            "include_glob": list(includes),
            "exclude_glob": list(excludes),
            "recursive": bool(recursive),
            "minimum_age_seconds": int(minimum_age_seconds),
            "preserve_markers": bool(preserve_markers),
        },
        "entries": entries,
    }
    plan_id = _plan_id(payload)
    confirmation = f"DELETE {root_kind.upper()} {plan_id[:12].upper()}"
    total_bytes = sum(entry["size"] for entry in entries)
    plan = {
        **payload,
        "plan_id": plan_id,
        "confirmation": confirmation,
        "summary": {
            "file_count": len(entries),
            "total_bytes": total_bytes,
            "total_gib": total_bytes / 1024**3,
            "skipped_count": len(skipped),
        },
        "skipped": skipped,
    }
    validate_storage_plan(plan)
    return plan


def storage_plan_report(plan: dict[str, Any]) -> dict[str, Any]:
    validate_storage_plan(plan)
    return {
        "format": plan["format"],
        "root": plan["root_kind"],
        "scope": plan["scope"],
        "plan_id": plan["plan_id"],
        "confirmation": plan["confirmation"],
        "summary": plan["summary"],
        "files": plan["entries"],
        "skipped": plan.get("skipped", []),
    }


def _receipt(
    plan: dict[str, Any],
    *,
    status: str,
    deleted: list[dict[str, Any]] | None = None,
    skipped: list[dict[str, str]] | None = None,
    removed_directories: list[str] | None = None,
) -> dict[str, Any]:
    deleted = deleted or []
    skipped = skipped or []
    return {
        "format": STORAGE_RECEIPT_FORMAT,
        "status": status,
        "root_kind": plan["root_kind"],
        "plan_id": plan["plan_id"],
        "completed_at_ns": time.time_ns(),
        "deleted": deleted,
        "deleted_count": len(deleted),
        "freed_bytes": sum(item["size"] for item in deleted),
        "skipped": skipped,
        "removed_directories": removed_directories or [],
    }


def apply_storage_plan(
    root: str | Path,
    plan: dict[str, Any],
    *,
    armed: bool,
    confirmation: str,
    remove_empty_directories: bool = True,
) -> dict[str, Any]:
    """Delete only unchanged files named by a validated plan."""

    validate_storage_plan(plan)
    base = Path(root).expanduser().resolve(strict=True)
    if not base.is_dir():
        raise NotADirectoryError(base)
    if not armed:
        return _receipt(plan, status="not_armed")
    if confirmation.strip() != plan["confirmation"]:
        raise ValueError(
            "confirmation does not match this plan; expected " + plan["confirmation"]
        )

    deleted: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    parent_candidates: set[Path] = set()
    for entry in plan["entries"]:
        relative = entry["relative_path"]
        try:
            path = resolve_planned_file(base, relative)
        except (OSError, ValueError) as exc:
            skipped.append({"relative_path": relative, "reason": f"unsafe_path:{exc}"})
            continue
        if not path.exists():
            skipped.append({"relative_path": relative, "reason": "missing"})
            continue
        if path.is_symlink() or not path.is_file():
            skipped.append({"relative_path": relative, "reason": "not_regular_file"})
            continue
        stat = path.stat()
        if stat.st_size != entry["size"] or stat.st_mtime_ns != entry["mtime_ns"]:
            skipped.append({"relative_path": relative, "reason": "changed_since_plan"})
            continue
        try:
            path.unlink()
        except OSError as exc:
            skipped.append({"relative_path": relative, "reason": f"delete_failed:{exc}"})
            continue
        deleted.append({"relative_path": relative, "size": entry["size"]})
        parent_candidates.add(path.parent)

    removed_directories: list[str] = []
    if remove_empty_directories:
        expanded: set[Path] = set()
        for parent in parent_candidates:
            current = parent
            while current != base and _is_within(base, current):
                expanded.add(current)
                current = current.parent
        for directory in sorted(expanded, key=lambda item: len(item.parts), reverse=True):
            try:
                directory.rmdir()
            except OSError:
                continue
            removed_directories.append(directory.relative_to(base).as_posix())

    if not plan["entries"]:
        status = "empty"
    elif skipped:
        status = "partial"
    else:
        status = "completed"
    return _receipt(
        plan,
        status=status,
        deleted=deleted,
        skipped=skipped,
        removed_directories=removed_directories,
    )
