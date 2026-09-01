"""Static contracts for H3 audiovisual carriers and timeline descriptors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict

NestedFactory = Callable[[tuple[Any, Any]], Any]


class _AVLatentOptional(TypedDict, total=False):
    """Optional AV carrier fields, expressed without Python 3.11-only typing APIs."""

    noise_mask: Any


class AVLatent(_AVLatentOptional):
    samples: Any


class AVWindowLayout(TypedDict):
    schema: str
    previous_frame_count: int
    window_start_frame: int
    window_end_frame: int
    window_frame_count: int
    overlap_frames: int
    extension_frames: int
    target_video_tokens: int
    target_audio_tokens: int
    overlap_video_tokens: int
    overlap_audio_tokens: int
    extension_video_tokens: int
    extension_audio_tokens: int
    layout_hash: str


class AVSpanDescriptor(TypedDict):
    timeline_origin_frame: int
    local_start_frame: int
    local_end_frame: int
    global_start_frame: int
    global_end_frame: int
    frame_count: int
    video_start_token: int
    video_end_token: int
    video_tokens: int
    audio_start_token: int
    audio_end_token: int
    audio_tokens: int
    video_spatial_shape: list[int]
    video_dtype: str
    audio_dtype: str
    video_device: str
    audio_device: str


class AVSpan(TypedDict):
    schema: str
    descriptor: AVSpanDescriptor
    descriptor_hash: str
    video: Any
    audio: Any
