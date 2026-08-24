"""Atomic persistence for nested H3 audiovisual latents."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any

from .h3 import get_av_streams


AV_LATENT_FORMAT = "cauce.h3-av-latent/1"


def _nested_tensor(video: Any, audio: Any):
    try:
        import comfy.nested_tensor  # type: ignore
    except ImportError as exc:  # pragma: no cover - requires ComfyUI
        raise RuntimeError("loading an H3 audiovisual latent requires ComfyUI") from exc
    return comfy.nested_tensor.NestedTensor((video, audio))


def save_av_latent_atomic(latent: dict[str, Any], path: str | Path) -> Path:
    """Atomically save the visual and structural-audio tensors."""

    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - ships with ComfyUI
        raise RuntimeError("safetensors is required to save H3 audiovisual latents") from exc

    video, audio = get_av_streams(latent)
    target = Path(path).expanduser().resolve()
    if target.suffix.lower() != ".safetensors":
        target = target.with_suffix(".safetensors")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".safetensors.tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(
            {
                "video": video.detach().cpu().contiguous(),
                "audio": audio.detach().cpu().contiguous(),
            },
            str(temporary),
            metadata={"format": AV_LATENT_FORMAT},
        )
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def load_av_latent(path: str | Path) -> dict[str, Any]:
    """Load a CAUCE H3 audiovisual latent onto CPU."""

    try:
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - ships with ComfyUI
        raise RuntimeError("safetensors is required to load H3 audiovisual latents") from exc

    target = Path(path).expanduser().resolve()
    data = load_file(str(target), device="cpu")
    if "video" not in data or "audio" not in data:
        raise ValueError("file does not contain H3 visual and structural-audio tensors")
    with safe_open(str(target), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("format") != AV_LATENT_FORMAT:
        raise ValueError("unrecognized CAUCE H3 audiovisual latent format")
    return {"samples": _nested_tensor(data["video"], data["audio"])}


def safe_output_path(root: str | Path, relative: str) -> Path:
    """Resolve a path strictly inside the configured ComfyUI output root."""

    base = Path(root).expanduser().resolve()
    raw = Path(str(relative).strip())
    if raw.is_absolute():
        raise ValueError("latent path must be relative to the ComfyUI output directory")
    target = (base / raw).resolve()
    if target != base and base not in target.parents:
        raise ValueError("latent path escapes the ComfyUI output directory")
    return target


def resolve_latest_or_indexed(
    root: str | Path,
    relative: str,
    *,
    artifact_index: int = 0,
) -> Path:
    """Resolve one explicit file or the newest matching latent in a folder."""

    target = safe_output_path(root, relative)
    if target.is_file():
        return target
    if not target.is_dir():
        raise FileNotFoundError(target)
    candidates = list(target.glob("*.safetensors"))
    if artifact_index > 0:
        ending = f"_{artifact_index:05d}.safetensors"
        candidates = [path for path in candidates if path.name.endswith(ending)]
    if not candidates:
        raise FileNotFoundError(f"no matching CAUCE H3 latent in {target}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)
