"""Fail-closed compatibility gates for official MiniMax H3 internals."""

from __future__ import annotations

import inspect


def require_arbitrary_h3_guides() -> None:
    """Require the released PackedLayout that accepts arbitrary AV guide spans."""

    try:
        from comfy.ldm.minimax.model import PackedLayout
    except ImportError as exc:  # pragma: no cover - requires ComfyUI
        raise RuntimeError("MiniMax H3 PackedLayout is unavailable in this ComfyUI build") from exc
    parameters = inspect.signature(PackedLayout.__init__).parameters
    if "frame_count" in parameters or not {"latent_t", "audio_t", "keyframes"} <= set(parameters):
        raise RuntimeError(
            "CAUCE H3 latent guides require a ComfyUI build with arbitrary-frame H3 AV guides"
        )
