"""Bounded, lossless export of native MoGe inference geometry."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

from .persistence import safe_output_path


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    import numpy as np
    return np.asarray(value)


def save_geometry_archive(geometry, root, filename_prefix, *, sample_index_start=0,
                          audit_stride=32):
    """Persist raw radial distance, validity and calibration, without normalization.

    The native inference node owns projection, masks and scale. This exporter
    computes only Euclidean point distance. It converts one sample at a time
    to CPU and never serializes the source images or creates a second inference.
    """
    import numpy as np

    if sample_index_start < 0 or audit_stride < 1:
        raise ValueError("sample index must be nonnegative and audit stride positive")
    points, masks, intrinsics = (geometry[k] for k in ("points", "mask", "intrinsics"))
    if len(points.shape) != 4 or points.shape[-1] != 3:
        raise ValueError("native points must have shape [B,H,W,3]")
    batch, height, width, _ = points.shape
    if tuple(masks.shape) != (batch, height, width):
        raise ValueError("native mask must match [B,H,W]")
    if tuple(intrinsics.shape) != (batch, 3, 3) or batch < 1:
        raise ValueError("native intrinsics must have shape [B,3,3] and batch cannot be empty")
    target = safe_output_path(root, filename_prefix + ".zip")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite geometry archive: {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".geometry-", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    manifest = {"format": "cauce.native-moge-geometry/1", "sample_count": int(batch),
                "sample_index_start": int(sample_index_start), "height": int(height),
                "width": int(width), "radial_definition": "Euclidean norm of native camera-space points",
                "scale": "Unmodified native inference scale; no per-sample normalization",
                "audit_stride": int(audit_stride), "samples": []}
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for i in range(batch):
                xyz = _numpy(points[i]).astype(np.float32, copy=False)
                radial = np.linalg.norm(xyz, axis=-1).astype(np.float32)
                mask = _numpy(masks[i]).astype(bool, copy=False)
                intrinsic = _numpy(intrinsics[i]).astype(np.float32, copy=False)
                index = int(sample_index_start + i)
                name = f"frame-{index:06d}.npz"
                buffer = io.BytesIO()
                np.savez_compressed(buffer, radial=radial, mask=mask, intrinsics=intrinsic,
                                    audit_points=xyz[::audit_stride, ::audit_stride],
                                    audit_stride=np.int32(audit_stride), sample_index=np.int64(index))
                payload = buffer.getvalue()
                archive.writestr(name, payload)
                manifest["samples"].append({"sample_index": index, "filename": name,
                    "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                    "valid_fraction": float(mask.mean()),
                    "finite_radial_fraction": float(np.isfinite(radial).mean())})
            archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
        # No other process should target the same task prefix; refuse an existing receipt.
        if target.exists():
            raise FileExistsError(target)
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return target, {"format": manifest["format"], "sample_count": int(batch),
                    "bytes": target.stat().st_size, "sha256": digest.hexdigest()}
