"""Compatibility facade for the split, typed H3 AV algebra.

The stable import path is retained for existing consumers.  Implementations are
partitioned by data responsibility under :mod:`cauce.av`.
"""

from .av.layout import (
    allocate_av_window_like,
    extract_h3_visual_stream,
    inspect_av_latent,
    plan_av_window,
    validate_av_window_layout,
)
from .av.masks import (
    apply_av_denoise_interval,
    apply_video_denoise_mask,
    clear_av_denoise_mask,
)
from .av.spatial import (
    densify_h3_video_tokens,
    expand_av_canvas,
    plan_h3_temporal_densification,
    replace_h3_video_stream,
    resize_h3_av_latent,
)
from .av.spans import (
    append_av_span,
    build_av_span_keyframes,
    extract_av_span,
    place_av_span,
    replace_av_span,
    split_av_latent,
    validate_av_span,
)
from .av.types import NestedFactory

__all__ = [
    "NestedFactory",
    "allocate_av_window_like",
    "append_av_span",
    "apply_av_denoise_interval",
    "apply_video_denoise_mask",
    "build_av_span_keyframes",
    "clear_av_denoise_mask",
    "densify_h3_video_tokens",
    "expand_av_canvas",
    "extract_av_span",
    "extract_h3_visual_stream",
    "inspect_av_latent",
    "place_av_span",
    "plan_av_window",
    "plan_h3_temporal_densification",
    "replace_av_span",
    "replace_h3_video_stream",
    "resize_h3_av_latent",
    "split_av_latent",
    "validate_av_span",
    "validate_av_window_layout",
]
