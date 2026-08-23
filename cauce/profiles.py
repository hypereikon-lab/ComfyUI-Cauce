"""Bounded execution profiles and non-mutating laboratory preflight checks."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
from typing import Any

from .contracts import PROFILE_SCHEMA


GIB = 1024**3

MODEL_FILES = {
    "fl2va": {
        "path": "diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
        "bytes": 20_958_205_608,
        "sha256": "12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993",
    },
    "ref2va": {
        "path": "diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
        "bytes": 20_958_205_608,
        "sha256": "f86f2f79ebd2d76eb8eeb46091e83982e6ff51d255747e7b16e92834b392b8e9",
    },
    "text_encoder": {
        "path": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "bytes": 15_687_142_551,
        "sha256": "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6",
    },
    "video_vae": {
        "path": "vae/minimax_h3_video_vae_fp16.safetensors",
        "bytes": 5_207_808_496,
        "sha256": "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
    },
    "audio_vae": {
        "path": "vae/minimax_h3_audio_vae_fp32.safetensors",
        "bytes": 605_254_808,
        "sha256": "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
    },
}


def _profile(name: str, family: str, width: int, height: int, tiled_vae: bool):
    model_keys = [family.lower(), "text_encoder", "video_vae", "audio_vae"]
    return {
        "schema": PROFILE_SCHEMA,
        "name": name,
        "family": family,
        "width": width,
        "height": height,
        "fps": 24,
        "precision": "fp8-pruned/nvfp4",
        "tiled_vae": tiled_vae,
        "vae_tile_size": 256,
        "vae_overlap": 64,
        "vae_temporal_size": 32,
        "vae_temporal_overlap": 8,
        "shift_video": 12.0,
        "shift_audio": 3.0,
        "model_keys": model_keys,
        "minimum_free_reserve_gib": 35,
        "minimum_vram_gib": 30,
    }


PROFILES = {
    "h3-5090-fl2va-640": _profile("h3-5090-fl2va-640", "FL2VA", 640, 640, False),
    "h3-5090-fl2va-768x512": _profile(
        "h3-5090-fl2va-768x512", "FL2VA", 768, 512, False
    ),
    "h3-5090-fl2va-800-tiled": _profile(
        "h3-5090-fl2va-800-tiled", "FL2VA", 800, 800, True
    ),
    "h3-5090-ref2va-448": _profile("h3-5090-ref2va-448", "Ref2VA", 448, 448, False),
    "h3-5090-ref2va-576x320": _profile(
        "h3-5090-ref2va-576x320", "Ref2VA", 576, 320, False
    ),
}


def get_profile(name: str) -> dict[str, Any]:
    if name not in PROFILES:
        raise ValueError(f"unknown CAUCE execution profile: {name}")
    return copy.deepcopy(PROFILES[name])


def model_manifest(profile: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(MODEL_FILES[key], key=key) for key in profile["model_keys"]]


def preflight(
    profile: dict[str, Any],
    models_root: str | Path,
    *,
    minimum_free_reserve_gib: float | None = None,
    inspect_torch: bool = True,
) -> dict[str, Any]:
    root = Path(models_root).expanduser().resolve()
    files = []
    for item in model_manifest(profile):
        path = (root / item["path"]).resolve()
        if root not in path.parents:
            raise ValueError("model manifest path escapes models_root")
        size = path.stat().st_size if path.exists() else 0
        files.append(
            {
                **item,
                "absolute_path": str(path),
                "present": path.is_file(),
                "actual_bytes": size,
                "size_ok": size == item["bytes"],
            }
        )
    disk_probe = root
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    disk = shutil.disk_usage(disk_probe)
    reserve = float(
        minimum_free_reserve_gib
        if minimum_free_reserve_gib is not None
        else profile.get("minimum_free_reserve_gib", 35)
    )
    runtime: dict[str, Any] = {}
    if inspect_torch:
        try:
            import torch

            runtime = {
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "device_capability": list(torch.cuda.get_device_capability(0))
                if torch.cuda.is_available()
                else None,
                "vram_bytes": torch.cuda.get_device_properties(0).total_memory
                if torch.cuda.is_available()
                else None,
            }
        except Exception as exc:  # diagnostics must remain usable
            runtime = {"error": f"{type(exc).__name__}: {exc}"}

    missing = [item["path"] for item in files if not item["present"]]
    malformed = [item["path"] for item in files if item["present"] and not item["size_ok"]]
    runtime_ready = True
    if inspect_torch:
        minimum_vram = float(profile.get("minimum_vram_gib", 0)) * GIB
        runtime_ready = bool(runtime.get("cuda_available")) and int(
            runtime.get("vram_bytes") or 0
        ) >= minimum_vram
    report = {
        "schema": "cauce.preflight/1",
        "profile": profile,
        "models_root": str(root),
        "models_root_exists": root.is_dir(),
        "files": files,
        "missing": missing,
        "unexpected_sizes": malformed,
        "disk_free_bytes": disk.free,
        "disk_free_gib": round(disk.free / GIB, 2),
        "reserve_gib": reserve,
        "runtime": runtime,
        "runtime_ready": runtime_ready,
    }
    report["ready"] = (
        root.is_dir()
        and not missing
        and not malformed
        and disk.free >= reserve * GIB
        and runtime_ready
    )
    return report


def format_preflight(report: dict[str, Any]) -> str:
    compact = {
        "ready": report["ready"],
        "profile": report["profile"]["name"],
        "models_root": report["models_root"],
        "models_root_exists": report["models_root_exists"],
        "disk_free_gib": report["disk_free_gib"],
        "reserve_gib": report["reserve_gib"],
        "missing": report["missing"],
        "unexpected_sizes": report["unexpected_sizes"],
        "runtime": report["runtime"],
        "runtime_ready": report["runtime_ready"],
    }
    return json.dumps(compact, ensure_ascii=False, indent=2)
