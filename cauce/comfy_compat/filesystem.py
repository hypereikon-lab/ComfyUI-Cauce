"""Adapter for the runtime-owned ComfyUI output directory."""

from __future__ import annotations

from pathlib import Path


def output_directory() -> Path:
    try:
        import folder_paths
    except ImportError as exc:  # pragma: no cover - requires ComfyUI
        raise RuntimeError("output persistence requires ComfyUI folder_paths") from exc
    getter = getattr(folder_paths, "get_output_directory", None)
    if not callable(getter):
        raise RuntimeError("this ComfyUI build does not expose an output directory")
    return Path(getter()).resolve()
