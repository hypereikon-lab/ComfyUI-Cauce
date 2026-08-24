"""H3 audiovisual-latent validation and temporal-mask capability checks."""

from __future__ import annotations

import importlib
import inspect
from typing import Any

from .timebase import visual_span_for_tokens


def get_av_streams(latent: dict[str, Any]) -> tuple[Any, Any]:
    """Return the visual and structural-audio streams from an H3 latent."""

    samples = latent.get("samples")
    if samples is None:
        raise ValueError("latent has no samples")
    if getattr(samples, "is_nested", False):
        streams = list(samples.unbind())
    elif isinstance(samples, (list, tuple)):
        streams = list(samples)
    else:
        raise ValueError("expected a nested MiniMax H3 audiovisual latent")
    if len(streams) < 2:
        raise ValueError("H3 latent must contain visual and structural-audio streams")
    video, audio = streams[0], streams[1]
    if getattr(video, "ndim", 0) != 5 or getattr(audio, "ndim", 0) != 4:
        raise ValueError("unexpected H3 audiovisual latent shapes")
    return video, audio


def pixel_frames_from_video_latent(video: Any) -> int:
    """Resolve the visible-frame span represented by an H3 visual stream."""

    return visual_span_for_tokens(int(video.shape[2]))


def h3_temporal_mask_capabilities() -> dict[str, Any]:
    """Inspect official ComfyUI support for model-aware per-row H3 masks.

    The probe intentionally checks the public execution seam between
    ``MiniMaxH3`` and ``MiniMaxH3Model``.  Helper functions used *inside* the
    model forward pass are implementation details and have already changed
    since native mask support landed; requiring one of those helpers would
    incorrectly reject a compatible current runtime.
    """

    problems: list[str] = []
    try:
        model_base = importlib.import_module("comfy.model_base")
        minimax_h3 = getattr(model_base, "MiniMaxH3")
    except (ImportError, AttributeError) as exc:  # pragma: no cover - inside ComfyUI
        minimax_h3 = None
        problems.append(f"MiniMaxH3 runtime unavailable ({type(exc).__name__}: {exc})")

    model_base_checks = {
        "token_grid_masks": callable(getattr(minimax_h3, "_token_grid_masks", None)),
        "denoise_mask_conds": callable(getattr(minimax_h3, "_denoise_mask_conds", None)),
        "scale_latent_inpaint": callable(getattr(minimax_h3, "scale_latent_inpaint", None)),
    }
    scale_latent = getattr(minimax_h3, "scale_latent_inpaint", None)
    try:
        scale_parameters = inspect.signature(scale_latent).parameters
        model_base_checks["scale_latent_receives_x_and_mask"] = {
            "x",
            "denoise_mask",
        }.issubset(scale_parameters)
    except (TypeError, ValueError):
        model_base_checks["scale_latent_receives_x_and_mask"] = False
    missing_mask_hooks = [name for name, ready in model_base_checks.items() if not ready]
    if missing_mask_hooks:
        problems.append("missing H3 denoise-mask hooks: " + ", ".join(missing_mask_hooks))

    try:
        engine = importlib.import_module("comfy.ldm.minimax.model")
        model = getattr(engine, "MiniMaxH3Model", None)
        forward = getattr(model, "forward", None)
        inner = getattr(model, "_forward", None)
        engine_checks = {
            "mask_row_values": callable(getattr(engine, "mask_row_values", None)),
            "forward_masks": callable(forward)
            and {"denoise_mask", "audio_denoise_mask"}.issubset(
                inspect.signature(forward).parameters
            ),
            "inner_masks": callable(inner)
            and {"denoise_mask", "audio_denoise_mask"}.issubset(
                inspect.signature(inner).parameters
            ),
        }
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        engine_checks = {
            "mask_row_values": False,
            "forward_masks": False,
            "inner_masks": False,
        }
        problems.append(f"H3 mask engine unavailable ({type(exc).__name__}: {exc})")

    missing_engine_hooks = [name for name, ready in engine_checks.items() if not ready]
    if missing_engine_hooks:
        problems.append("incomplete H3 per-row mask engine: " + ", ".join(missing_engine_hooks))

    return {
        "schema": "cauce.h3-temporal-mask-capabilities/2",
        "ready": not problems,
        "per_token_denoise_mask": not missing_mask_hooks and not missing_engine_hooks,
        "per_row_denoise_mask": not missing_mask_hooks and not missing_engine_hooks,
        "model_base": model_base_checks,
        "missing_mask_hooks": missing_mask_hooks,
        "mask_engine": engine_checks,
        "missing_engine_hooks": missing_engine_hooks,
        "problems": problems,
    }


def require_h3_temporal_mask_runtime() -> dict[str, Any]:
    """Fail closed when H3 cannot preserve unmasked temporal rows."""

    capabilities = h3_temporal_mask_capabilities()
    if not capabilities["ready"]:
        detail = "; ".join(capabilities["problems"])
        raise RuntimeError(
            "CAUCE temporal inpainting requires official per-token MiniMax H3 "
            f"denoise masks; the current runtime is unsafe for this operation: {detail}"
        )
    return capabilities
