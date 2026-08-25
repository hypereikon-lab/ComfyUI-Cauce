"""Deterministic preparation and assembly for native H3 guide bridges.

This module never alters H3 latents or sampling.  It selects two decoded guide
clips for official ``MiniMaxH3AddGuide`` nodes and, after normal H3 generation,
accepts only the generated interval between those protected guides.
"""

from __future__ import annotations

from typing import Any

from .contracts import BRIDGE_SCHEMA, content_hash
from .timebase import H3_FPS, H3_TRAINED_MIN_FRAMES, as_fraction, is_h3_frame_count


H3_MAX_FRAMES = 362
VALID_GUIDE_FRAMES = tuple(range(5, 108, 17))
VALID_TARGET_FRAMES = tuple(range(H3_TRAINED_MIN_FRAMES, H3_MAX_FRAMES + 1, 17))


def _frame_count(value: Any, name: str) -> int:
    try:
        frames = int(value.shape[0])
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError(f"{name} must be an IMAGE batch") from exc
    if frames < 1:
        raise ValueError(f"{name} cannot be empty")
    return frames


def _image_geometry(value: Any, name: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(item) for item in value.shape)
    except (AttributeError, TypeError) as exc:
        raise ValueError(f"{name} must be an IMAGE batch") from exc
    if len(shape) != 4:
        raise ValueError(f"{name} must have shape [frames,height,width,channels]")
    return shape[1:]


def _copy(value: Any):
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    copy = getattr(value, "copy", None)
    return copy() if callable(copy) else value


def _concatenate(values: tuple[Any, ...]):
    first = values[0]
    try:
        import torch

        if isinstance(first, torch.Tensor):
            return torch.cat(values, dim=0)
    except ImportError:  # pragma: no cover - torch ships with ComfyUI
        pass

    try:
        import numpy as np

        if isinstance(first, np.ndarray):
            return np.concatenate(values, axis=0)
    except ImportError:  # pragma: no cover - NumPy ships with ComfyUI
        pass
    raise TypeError("IMAGE batches must be PyTorch tensors or NumPy arrays")


def plan_native_guide_bridge(
    left_frame_count: int,
    right_frame_count: int,
    *,
    guide_frames: int = 22,
    target_frames: int = 124,
    fps: float = 24.0,
) -> dict[str, Any]:
    """Plan a fresh H3 target constrained by source-tail and source-head guides."""

    left_frames = int(left_frame_count)
    right_frames = int(right_frame_count)
    guide = int(guide_frames)
    target = int(target_frames)
    if as_fraction(fps) != H3_FPS:
        raise ValueError("native H3 guide bridges require 24 fps source batches")
    if guide not in VALID_GUIDE_FRAMES:
        raise ValueError(f"guide_frames must be one of {VALID_GUIDE_FRAMES}")
    if target not in VALID_TARGET_FRAMES or not is_h3_frame_count(target):
        raise ValueError(f"target_frames must be one of {VALID_TARGET_FRAMES}")
    if left_frames < guide or right_frames < guide:
        raise ValueError("each source batch must contain at least guide_frames frames")
    if target <= guide * 2:
        raise ValueError("target must leave a non-empty generated center between guides")

    right_guide_frame_idx = target - guide
    center_frames = target - 2 * guide
    payload: dict[str, Any] = {
        "schema": BRIDGE_SCHEMA,
        "fps": 24,
        "left_source_frames": left_frames,
        "right_source_frames": right_frames,
        "guide_frames": guide,
        "target_frames": target,
        "left_source_range": [left_frames - guide, left_frames],
        "right_source_range": [0, guide],
        "left_guide_frame_idx": 0,
        "right_guide_frame_idx": right_guide_frame_idx,
        "generated_center_range": [guide, right_guide_frame_idx],
        "generated_center_frames": center_frames,
        "assembled_frames": left_frames + center_frames + right_frames,
        "conditioning_owner": "official MiniMaxH3AddGuide nodes",
        "sampling_owner": "official ComfyUI H3 workflow",
        "assembly": "complete left source + generated center + complete right source",
    }
    payload["plan_hash"] = content_hash(payload)
    return payload


def validate_bridge_plan(plan: dict[str, Any]) -> None:
    if not isinstance(plan, dict) or plan.get("schema") != BRIDGE_SCHEMA:
        raise ValueError(f"bridge plan must use schema {BRIDGE_SCHEMA}")
    supplied_hash = plan.get("plan_hash")
    unhashed = {key: value for key, value in plan.items() if key != "plan_hash"}
    if supplied_hash != content_hash(unhashed):
        raise ValueError("bridge plan hash does not match its contents")


def extract_native_guide_bridge_sources(
    left_images: Any,
    right_images: Any,
    plan: dict[str, Any],
) -> tuple[Any, Any]:
    """Extract the exact clips intended for two official H3 AddGuide nodes."""

    validate_bridge_plan(plan)
    left_count = _frame_count(left_images, "left_images")
    right_count = _frame_count(right_images, "right_images")
    if left_count != int(plan["left_source_frames"]):
        raise ValueError("left source length differs from the bridge plan")
    if right_count != int(plan["right_source_frames"]):
        raise ValueError("right source length differs from the bridge plan")
    if _image_geometry(left_images, "left_images") != _image_geometry(
        right_images, "right_images"
    ):
        raise ValueError("left and right source batches must have matching geometry")
    left_start, left_end = map(int, plan["left_source_range"])
    right_start, right_end = map(int, plan["right_source_range"])
    return _copy(left_images[left_start:left_end]), _copy(right_images[right_start:right_end])


def apply_native_guide_bridge(
    left_images: Any,
    right_images: Any,
    generated_target: Any,
    plan: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    """Accept only the generated center and insert it between complete sources."""

    extract_native_guide_bridge_sources(left_images, right_images, plan)
    generated_count = _frame_count(generated_target, "generated_target")
    if generated_count != int(plan["target_frames"]):
        raise ValueError(
            f"generated target has {generated_count} frames; expected {plan['target_frames']}"
        )
    expected_geometry = _image_geometry(left_images, "left_images")
    if _image_geometry(generated_target, "generated_target") != expected_geometry:
        raise ValueError("generated target and sources must have matching geometry")
    center_start, center_end = map(int, plan["generated_center_range"])
    center = _copy(generated_target[center_start:center_end])
    joined = _concatenate((left_images, center, right_images))
    report = {
        "schema": "cauce.h3-native-guide-bridge-report/1",
        "plan_hash": plan["plan_hash"],
        "accepted_generated_range": [center_start, center_end],
        "accepted_generated_frames": int(center.shape[0]),
        "assembled_frames": int(joined.shape[0]),
        "quality_status": "requires_visual_validation",
    }
    return joined, center, report
