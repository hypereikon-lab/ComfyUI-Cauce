"""Atomic CAUCE artifact and nested audiovisual latent persistence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .contracts import canonical_json
from .h3 import get_av_streams


AV_LATENT_FORMAT = "cauce.h3-av-latent/1"


def write_json_atomic(path: str | Path, value: Any) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def read_json(path: str | Path) -> Any:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _nested_tensor(video: Any, audio: Any):
    try:
        import comfy.nested_tensor  # type: ignore
    except ImportError as exc:  # pragma: no cover - requires ComfyUI
        raise RuntimeError("loading a CAUCE AV latent requires ComfyUI") from exc
    return comfy.nested_tensor.NestedTensor((video, audio))


def save_av_latent_atomic(
    latent: dict[str, Any],
    path: str | Path,
    *,
    receipt: dict[str, Any] | None = None,
) -> Path:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - safetensors ships with ComfyUI
        raise RuntimeError("safetensors is required to save CAUCE AV latents") from exc

    video, audio = get_av_streams(latent)
    target = Path(path).expanduser().resolve()
    if target.suffix.lower() != ".safetensors":
        target = target.with_suffix(".safetensors")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".safetensors.tmp", dir=target.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    metadata = {"format": AV_LATENT_FORMAT}
    if receipt is not None:
        metadata["receipt"] = canonical_json(receipt)
    try:
        save_file(
            {
                "video": video.detach().cpu().contiguous(),
                "audio": audio.detach().cpu().contiguous(),
            },
            str(temporary),
            metadata=metadata,
        )
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def load_av_latent(path: str | Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - safetensors ships with ComfyUI
        raise RuntimeError("safetensors is required to load CAUCE AV latents") from exc

    target = Path(path).expanduser().resolve()
    data = load_file(str(target), device="cpu")
    if "video" not in data or "audio" not in data:
        raise ValueError("file does not contain CAUCE video/audio latent streams")
    receipt = None
    with safe_open(str(target), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("format") not in {AV_LATENT_FORMAT, "h3_motion_context_av_v1"}:
        raise ValueError("unrecognized audiovisual latent format")
    if metadata.get("receipt"):
        receipt = json.loads(metadata["receipt"])
    return {"samples": _nested_tensor(data["video"], data["audio"])}, receipt


def safe_output_path(root: str | Path, relative: str, suffix: str = "") -> Path:
    base = Path(root).expanduser().resolve()
    raw = Path(str(relative).strip())
    if raw.is_absolute():
        target = raw.resolve()
    else:
        target = (base / raw).resolve()
    if target != base and base not in target.parents:
        raise ValueError("artifact path escapes the configured output directory")
    if suffix and target.suffix.lower() != suffix.lower():
        target = target.with_suffix(suffix)
    return target


def resolve_latest_or_indexed(
    root: str | Path,
    relative: str,
    *,
    artifact_index: int = 0,
) -> Path:
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
        raise FileNotFoundError(f"no matching CAUCE AV latent in {target}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)
