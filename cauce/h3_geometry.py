"""Experimental geometry-coordinate transforms for MiniMax H3.

This module deliberately changes only token coordinates.  It does not claim to
turn H3 into a calibrated fisheye model, and it never changes pixels, latents,
weights, prompts, or the sampling schedule.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any


DOMEMASTER_PROFILE = "equidistant_180_ray_xy"
DOMEMASTER_WRAPPER_KEY = "cauce_h3_domemaster_coordinate_warp"


def equidistant_180_ray_xy(u: float, v: float) -> tuple[float, float, float]:
    """Map an equidistant 180-degree domemaster sample to a unit camera ray.

    ``u`` and ``v`` are normalized disc coordinates.  The returned tuple is
    ``(ray_x, ray_y, ray_z)`` on the front hemisphere.  Samples outside the
    unit disc are rejected because they do not represent a camera ray.
    """

    x = float(u)
    y = float(v)
    radius = math.hypot(x, y)
    if radius > 1.0 + 1e-12:
        raise ValueError("equidistant domemaster coordinates must lie inside the unit disc")
    if radius <= 1e-12:
        return 0.0, 0.0, 1.0
    theta = min(1.0, radius) * (math.pi / 2.0)
    radial = math.sin(theta) / radius
    return x * radial, y * radial, math.cos(theta)


def _stock_axis(dim: int, patch: int, sqrt_area: float) -> list[float]:
    """Match MiniMax H3's area-normalized ``_axis_from_sqrt_area`` exactly."""

    size = int(dim)
    step = int(patch)
    if size < step or size % step:
        raise ValueError("H3 spatial latent dimensions must be divisible by the DiT patch")
    ratio = size / float(sqrt_area)
    count = size // step
    return [
        (index * (ratio / count) + (1.0 - ratio) / 2.0) * 32.0
        for index in range(count)
    ]


def domemaster_coordinate_rows(
    latent_height: int,
    latent_width: int,
    *,
    strength: float = 1.0,
    outside_disc: str = "stock",
) -> list[tuple[float, float]]:
    """Return H3 ``(h,w)`` rows with a front-hemisphere coordinate bias.

    At strength zero this is bit-equivalent to H3's stock spatial grid.  At
    strength one, samples inside the domemaster disc use the x/y components of
    their equidistant camera ray, scaled back to the stock coordinate extent.
    The z component is implicit because the front-hemisphere ray is uniquely
    determined by x/y.  Outside-disc rows remain stock by default so the patch
    introduces no coordinate collisions in the black support region.
    """

    height = int(latent_height)
    width = int(latent_width)
    amount = float(strength)
    if height != width:
        raise ValueError("domemaster coordinate warping requires a square H3 latent")
    if not 0.0 <= amount <= 1.0:
        raise ValueError("domemaster coordinate strength must lie in [0, 1]")
    if outside_disc not in {"stock", "rim"}:
        raise ValueError("outside_disc must be stock or rim")

    sqrt_area = math.sqrt(height * width)
    h_axis = _stock_axis(height, 2, sqrt_area)
    w_axis = _stock_axis(width, 2, sqrt_area)
    h_center = (h_axis[0] + h_axis[-1]) / 2.0
    w_center = (w_axis[0] + w_axis[-1]) / 2.0
    h_radius = (h_axis[-1] - h_axis[0]) / 2.0
    w_radius = (w_axis[-1] - w_axis[0]) / 2.0
    if h_radius <= 0.0 or w_radius <= 0.0:
        raise ValueError("domemaster coordinate warping needs at least two rows per axis")

    rows: list[tuple[float, float]] = []
    for stock_h in h_axis:
        v = (stock_h - h_center) / h_radius
        for stock_w in w_axis:
            u = (stock_w - w_center) / w_radius
            radius = math.hypot(u, v)
            if radius <= 1.0 + 1e-12:
                ray_x, ray_y, _ = equidistant_180_ray_xy(u, v)
                target_h = h_center + h_radius * ray_y
                target_w = w_center + w_radius * ray_x
            elif outside_disc == "rim":
                target_h = h_center + h_radius * (v / radius)
                target_w = w_center + w_radius * (u / radius)
            else:
                target_h, target_w = stock_h, stock_w
            rows.append(
                (
                    stock_h + amount * (target_h - stock_h),
                    stock_w + amount * (target_w - stock_w),
                )
            )
    return rows


def _segment_kinds(segments: Iterable[Sequence[Any]]) -> tuple[str, ...]:
    return tuple(str(segment[2]) for segment in segments)


def warp_h3_layout_position_ids(
    position_ids: Any,
    segments: Iterable[Sequence[Any]],
    *,
    latent_height: int,
    latent_width: int,
    strength: float,
    include_keyframes: bool,
    outside_disc: str = "stock",
) -> tuple[Any, dict[str, Any]]:
    """Clone and warp selected ``PackedLayout.position_ids`` spatial rows.

    The tensor operations are intentionally duck-typed so importing CAUCE does
    not import PyTorch outside ComfyUI.  Target-video rows are always selected;
    FL2VA keyframe rows are selected only when requested.  Ref2VA reference
    rows are never altered because their projection is not necessarily the
    target domemaster projection.
    """

    segment_list = [tuple(segment) for segment in segments]
    selected = {"video"}
    if include_keyframes:
        selected.add("cond")
    rows = domemaster_coordinate_rows(
        latent_height,
        latent_width,
        strength=strength,
        outside_disc=outside_disc,
    )
    frame_rows = len(rows)
    result = position_ids.clone()
    row_tensor = result.new_tensor(rows)
    changed_segments: list[dict[str, Any]] = []
    for start, stop, kind in segment_list:
        start_i = int(start)
        stop_i = int(stop)
        kind_s = str(kind)
        if kind_s not in selected:
            continue
        count = stop_i - start_i
        if count % frame_rows:
            raise ValueError(
                f"H3 {kind_s} segment has {count} rows, not a multiple of {frame_rows}"
            )
        repeats = count // frame_rows
        result[start_i:stop_i, 1:] = row_tensor.repeat((repeats, 1))
        changed_segments.append(
            {"kind": kind_s, "row_range": [start_i, stop_i], "frames": repeats}
        )
    report = {
        "schema": "cauce.h3-domemaster-coordinate-report/1",
        "profile": DOMEMASTER_PROFILE,
        "strength": float(strength),
        "outside_disc": outside_disc,
        "latent_shape": [int(latent_height), int(latent_width)],
        "dit_spatial_rows": frame_rows,
        "include_keyframes": bool(include_keyframes),
        "layout_segment_kinds": list(_segment_kinds(segment_list)),
        "changed_segments": changed_segments,
    }
    return result, report


class H3DomemasterCoordinatePatch:
    """Comfy diffusion-model wrapper that temporarily patches H3 layout RoPE."""

    def __init__(self, strength: float, include_keyframes: bool, outside_disc: str):
        self.strength = float(strength)
        self.include_keyframes = bool(include_keyframes)
        self.outside_disc = str(outside_disc)
        self.last_report: dict[str, Any] | None = None

    def __call__(
        self,
        executor,
        x,
        timestep,
        context,
        transformer_options=None,
        **kwargs,
    ):
        # The explicit zero-strength control must be bit-identical to stock H3:
        # do not reconstruct or temporarily replace any coordinates.
        if self.strength == 0.0:
            self.last_report = {
                "schema": "cauce.h3-domemaster-coordinate-report/1",
                "profile": DOMEMASTER_PROFILE,
                "strength": 0.0,
                "bypassed_bit_exactly": True,
                "changed_segments": [],
            }
            return executor(
                x,
                timestep,
                context,
                transformer_options or {},
                **kwargs,
            )
        payload = kwargs.get("minimax_payload") or {}
        layout = payload.get("layout")
        if layout is None:
            raise ValueError(
                "CAUCE domemaster coordinates require MiniMax H3's prebuilt PackedLayout"
            )
        video = x[0]
        latent_height = int(video.shape[3])
        latent_width = int(video.shape[4])
        original = layout.position_ids
        warped, report = warp_h3_layout_position_ids(
            original,
            layout.segments,
            latent_height=latent_height,
            latent_width=latent_width,
            strength=self.strength,
            include_keyframes=self.include_keyframes,
            outside_disc=self.outside_disc,
        )
        self.last_report = report
        layout.position_ids = warped
        try:
            return executor(
                x,
                timestep,
                context,
                transformer_options or {},
                **kwargs,
            )
        finally:
            layout.position_ids = original


__all__ = [
    "DOMEMASTER_PROFILE",
    "DOMEMASTER_WRAPPER_KEY",
    "H3DomemasterCoordinatePatch",
    "domemaster_coordinate_rows",
    "equidistant_180_ray_xy",
    "warp_h3_layout_position_ids",
]
