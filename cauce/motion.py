"""Coordinate-map algebra and deterministic image-space motion.

The public map contract is deliberately semantic-free.  A map stores an
inverse sampling grid (target -> source), validity, and an exact temporal
domain.  Affines, analytic maps, advection, depth reprojection, optical flow,
and future simulations can therefore be composed before media is sampled.

Map construction uses NumPy so its mathematics remains testable outside a
ComfyUI/PyTorch runtime.  IMAGE sampling is lazy-imported and stays on the
tensor's device.  H3 conditioning and sampling deliberately remain upstream.
"""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any

import numpy as np

MOTION_MAP_SCHEMA = "cauce.motion-map/1"
VECTOR_FIELD_SCHEMA = "cauce.vector-field/1"
MAP_DIRECTION = "target_to_source"
COORDINATE_SYSTEM = "pytorch_normalized_align_corners_false"
EASINGS = ("linear", "smoothstep", "cosine", "sine_loop")
PADDING_MODES = ("zeros", "border", "reflection")
ANALYTIC_MAPS = ("swirl", "pinch", "wave", "radial_wave", "tunnel", "kaleidoscope")
VECTOR_FIELDS = ("uniform", "rotation", "radial", "vortex", "curl_sine", "wave")
INTEGRATORS = ("euler", "rk2", "rk4")


def _dimensions(frames: int, height: int, width: int) -> tuple[int, int, int]:
    values = (int(frames), int(height), int(width))
    if min(values) < 1:
        raise ValueError("map frames, height, and width must be positive")
    if values[0] * values[1] * values[2] > 64_000_000:
        raise ValueError("motion map exceeds the 64M-cell safety limit")
    return values


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def identity_grid(frames: int, height: int, width: int) -> np.ndarray:
    frames, height, width = _dimensions(frames, height, width)
    x = (2.0 * (np.arange(width, dtype=np.float32) + 0.5) / width) - 1.0
    y = (2.0 * (np.arange(height, dtype=np.float32) + 0.5) / height) - 1.0
    xx, yy = np.meshgrid(x, y, indexing="xy")
    grid = np.stack((xx, yy), axis=-1)
    return np.broadcast_to(grid, (frames, height, width, 2)).copy()


def _motion_map(
    grid: Any,
    validity: Any | None = None,
    *,
    fps: float = 24.0,
    time_domain: str = "visible_frames",
    operation: str = "identity",
    parameters: dict[str, Any] | None = None,
    provenance: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    grid = np.asarray(grid, dtype=np.float32)
    if grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError("motion grid must have shape [frames,height,width,2]")
    frames, height, width = _dimensions(*grid.shape[:3])
    if not np.isfinite(grid).all():
        raise ValueError("motion grid contains non-finite coordinates")
    if validity is None:
        validity = validity_from_grid(grid)
    validity = np.asarray(validity, dtype=np.float32)
    if validity.shape != (frames, height, width):
        raise ValueError("map validity must have shape [frames,height,width]")
    validity = np.clip(validity, 0.0, 1.0)
    if float(fps) <= 0:
        raise ValueError("map fps must be positive")
    metadata = {
        "schema": MOTION_MAP_SCHEMA,
        "direction": MAP_DIRECTION,
        "coordinate_system": COORDINATE_SYSTEM,
        "align_corners": False,
        "frames": frames,
        "height": height,
        "width": width,
        "fps": float(fps),
        "time_domain": str(time_domain),
        "operation": str(operation),
        "parameters": copy.deepcopy(parameters or {}),
        "provenance": copy.deepcopy(provenance or []),
    }
    metadata["tensor_hash"] = _array_hash(grid, validity)
    return {**metadata, "grid": grid, "validity": validity}


def validate_motion_map(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != MOTION_MAP_SCHEMA:
        raise ValueError(f"motion map schema must be {MOTION_MAP_SCHEMA}")
    if value.get("direction") != MAP_DIRECTION:
        raise ValueError("motion map must use target-to-source sampling")
    if value.get("coordinate_system") != COORDINATE_SYSTEM:
        raise ValueError("motion map uses an unsupported coordinate system")
    grid = np.asarray(value.get("grid"), dtype=np.float32)
    validity = np.asarray(value.get("validity"), dtype=np.float32)
    expected = (int(value["frames"]), int(value["height"]), int(value["width"]))
    if grid.shape != expected + (2,) or validity.shape != expected:
        raise ValueError("motion map tensor geometry disagrees with its metadata")
    if not np.isfinite(grid).all() or not np.isfinite(validity).all():
        raise ValueError("motion map contains non-finite values")
    return value


def motion_map_report(value: dict[str, Any]) -> dict[str, Any]:
    validate_motion_map(value)
    grid = np.asarray(value["grid"])
    validity = np.asarray(value["validity"])
    identity = identity_grid(*grid.shape[:3])
    displacement = np.linalg.norm(grid - identity, axis=-1)
    return {
        "schema": "cauce.motion-map-report/1",
        "operation": value["operation"],
        "frames": value["frames"],
        "height": value["height"],
        "width": value["width"],
        "fps": value["fps"],
        "time_domain": value["time_domain"],
        "valid_fraction": float(validity.mean()),
        "maximum_normalized_displacement": float(displacement.max()),
        "mean_normalized_displacement": float(displacement.mean()),
        "tensor_hash": value["tensor_hash"],
        "parameters": copy.deepcopy(value.get("parameters", {})),
    }


def _progress(frames: int, easing: str) -> np.ndarray:
    if easing not in EASINGS:
        raise ValueError(f"easing must be one of {', '.join(EASINGS)}")
    if frames == 1:
        return np.zeros(1, dtype=np.float32)
    x = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    if easing == "linear":
        return x
    if easing == "smoothstep":
        return x * x * (3.0 - 2.0 * x)
    if easing == "cosine":
        return 0.5 - 0.5 * np.cos(np.pi * x)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * x)


def _lerp(start: float, end: float, progress: np.ndarray) -> np.ndarray:
    return float(start) + (float(end) - float(start)) * progress


def _log_lerp(start: float, end: float, progress: np.ndarray) -> np.ndarray:
    if float(start) <= 0 or float(end) <= 0:
        raise ValueError("scale values must be positive")
    return np.exp(_lerp(math.log(float(start)), math.log(float(end)), progress))


def _sample_numpy(image: np.ndarray, grid: np.ndarray, padding_mode: str = "border") -> np.ndarray:
    """Bilinear sample one HWC image with an align_corners=False grid."""

    if padding_mode not in PADDING_MODES:
        raise ValueError(f"padding mode must be one of {', '.join(PADDING_MODES)}")
    image = np.asarray(image)
    if image.ndim == 2:
        image = image[..., None]
    height, width, channels = image.shape
    gx, gy = grid[..., 0], grid[..., 1]
    x = ((gx + 1.0) * width - 1.0) * 0.5
    y = ((gy + 1.0) * height - 1.0) * 0.5
    if padding_mode == "border":
        x = np.clip(x, 0.0, max(0.0, width - 1.0))
        y = np.clip(y, 0.0, max(0.0, height - 1.0))
    elif padding_mode == "reflection":
        if width > 1:
            x = np.abs((x + width - 1.0) % (2.0 * (width - 1.0)) - (width - 1.0))
        else:
            x = np.zeros_like(x)
        if height > 1:
            y = np.abs((y + height - 1.0) % (2.0 * (height - 1.0)) - (height - 1.0))
        else:
            y = np.zeros_like(y)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = x - x0, y - y0

    def gather(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        values = image[np.clip(iy, 0, height - 1), np.clip(ix, 0, width - 1)]
        if padding_mode == "zeros":
            values = values * valid[..., None]
        return values

    result = (
        gather(x0, y0) * ((1.0 - wx) * (1.0 - wy))[..., None]
        + gather(x1, y0) * (wx * (1.0 - wy))[..., None]
        + gather(x0, y1) * ((1.0 - wx) * wy)[..., None]
        + gather(x1, y1) * (wx * wy)[..., None]
    )
    return result.reshape(grid.shape[:2] + (channels,))


def validity_from_grid(grid: np.ndarray) -> np.ndarray:
    frames, height, width = grid.shape[:3]
    ones = np.ones((height, width, 1), dtype=np.float32)
    return np.stack(
        [_sample_numpy(ones, grid[index], "zeros")[..., 0] for index in range(frames)],
        axis=0,
    ).astype(np.float32)


def affine_motion_map(
    frames: int,
    height: int,
    width: int,
    *,
    translate_x_start: float = 0.0,
    translate_x_end: float = 0.0,
    translate_y_start: float = 0.0,
    translate_y_end: float = 0.0,
    scale_start: float = 1.0,
    scale_end: float = 1.0,
    scale_y_start: float | None = None,
    scale_y_end: float | None = None,
    rotation_start: float = 0.0,
    rotation_end: float = 0.0,
    pivot_x_percent: float = 50.0,
    pivot_y_percent: float = 50.0,
    easing: str = "smoothstep",
    fps: float = 24.0,
) -> dict[str, Any]:
    frames, height, width = _dimensions(frames, height, width)
    progress = _progress(frames, easing)
    tx = 2.0 * _lerp(translate_x_start, translate_x_end, progress) / 100.0
    ty = 2.0 * _lerp(translate_y_start, translate_y_end, progress) / 100.0
    sx = _log_lerp(scale_start, scale_end, progress)
    sy = _log_lerp(
        scale_start if scale_y_start is None else scale_y_start,
        scale_end if scale_y_end is None else scale_y_end,
        progress,
    )
    angle = np.deg2rad(_lerp(rotation_start, rotation_end, progress))
    px = 2.0 * float(pivot_x_percent) / 100.0 - 1.0
    py = 2.0 * float(pivot_y_percent) / 100.0 - 1.0
    base = identity_grid(1, height, width)[0]
    homogeneous = np.concatenate(
        (base, np.ones((height, width, 1), dtype=np.float32)), axis=-1
    )
    grids = []
    for index in range(frames):
        cosine, sine = math.cos(float(angle[index])), math.sin(float(angle[index]))
        transform = np.array(
            [
                [cosine * sx[index], -sine * sy[index], tx[index] + px],
                [sine * sx[index], cosine * sy[index], ty[index] + py],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ) @ np.array(
            [[1.0, 0.0, -px], [0.0, 1.0, -py], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        inverse = np.linalg.inv(transform)
        sampled = homogeneous @ inverse.T
        grids.append((sampled[..., :2] / sampled[..., 2:3]).astype(np.float32))
    grid = np.stack(grids, axis=0)
    parameters = {
        "translate_x_percent": [translate_x_start, translate_x_end],
        "translate_y_percent": [translate_y_start, translate_y_end],
        "scale_x": [scale_start, scale_end],
        "scale_y": [scale_y_start if scale_y_start is not None else scale_start, scale_y_end if scale_y_end is not None else scale_end],
        "rotation_degrees": [rotation_start, rotation_end],
        "pivot_percent": [pivot_x_percent, pivot_y_percent],
        "easing": easing,
    }
    return _motion_map(grid, fps=fps, operation="affine", parameters=parameters)


def _homography_from_corners(source: np.ndarray) -> np.ndarray:
    """Solve a target-square -> source-quadrilateral homography."""

    target = np.asarray(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        dtype=np.float64,
    )
    source = np.asarray(source, dtype=np.float64)
    if source.shape != (4, 2):
        raise ValueError("corner coordinates must have shape [4,2]")
    rows, values = [], []
    for (x, y), (u, v) in zip(target, source):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.extend((u, v))
    coefficients = np.linalg.solve(
        np.asarray(rows, dtype=np.float64), np.asarray(values, dtype=np.float64)
    )
    return np.append(coefficients, 1.0).reshape(3, 3)


def perspective_motion_map(
    frames: int,
    height: int,
    width: int,
    *,
    top_left_x_end: float = 0.0,
    top_left_y_end: float = 0.0,
    top_right_x_end: float = 0.0,
    top_right_y_end: float = 0.0,
    bottom_right_x_end: float = 0.0,
    bottom_right_y_end: float = 0.0,
    bottom_left_x_end: float = 0.0,
    bottom_left_y_end: float = 0.0,
    easing: str = "smoothstep",
    fps: float = 24.0,
) -> dict[str, Any]:
    """Build a projective pullback from percentage offsets at four corners."""

    frames, height, width = _dimensions(frames, height, width)
    progress = _progress(frames, easing)
    base = identity_grid(1, height, width)[0]
    homogeneous = np.concatenate(
        (base, np.ones((height, width, 1), dtype=np.float32)), axis=-1
    )
    offsets = np.asarray(
        [
            [top_left_x_end, top_left_y_end],
            [top_right_x_end, top_right_y_end],
            [bottom_right_x_end, bottom_right_y_end],
            [bottom_left_x_end, bottom_left_y_end],
        ],
        dtype=np.float64,
    )
    target_corners = np.asarray(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        dtype=np.float64,
    )
    grids = []
    for fraction in progress:
        corners = target_corners + 2.0 * offsets * float(fraction) / 100.0
        homography = _homography_from_corners(corners)
        sampled = homogeneous @ homography.T
        denominator = sampled[..., 2:3]
        if np.any(np.abs(denominator) < 1e-8):
            raise ValueError("corner offsets create a singular projective map")
        grids.append((sampled[..., :2] / denominator).astype(np.float32))
    return _motion_map(
        np.stack(grids),
        fps=fps,
        operation="perspective_corner_pin",
        parameters={"corner_offsets_percent": offsets.tolist(), "easing": easing},
    )


def analytic_motion_map(
    frames: int,
    height: int,
    width: int,
    *,
    mode: str,
    amount_start: float = 0.0,
    amount_end: float = 30.0,
    frequency: float = 2.0,
    phase_cycles: float = 0.0,
    sides: int = 6,
    easing: str = "smoothstep",
    fps: float = 24.0,
) -> dict[str, Any]:
    if mode not in ANALYTIC_MAPS:
        raise ValueError(f"analytic map must be one of {', '.join(ANALYTIC_MAPS)}")
    frames, height, width = _dimensions(frames, height, width)
    base = identity_grid(1, height, width)[0]
    x, y = base[..., 0], base[..., 1]
    radius = np.sqrt(x * x + y * y)
    angle = np.arctan2(y, x)
    progress = _progress(frames, easing)
    amounts = _lerp(amount_start, amount_end, progress)
    grids = []
    for index, amount in enumerate(amounts):
        phase = 2.0 * np.pi * (phase_cycles * progress[index])
        if mode == "swirl":
            source_angle = angle - np.deg2rad(amount) * np.clip(1.0 - radius / math.sqrt(2.0), 0.0, 1.0) ** max(0.1, frequency)
            sx, sy = radius * np.cos(source_angle), radius * np.sin(source_angle)
        elif mode == "pinch":
            exponent = math.exp(float(amount) / 100.0)
            source_radius = np.power(np.clip(radius, 1e-6, None), exponent)
            sx, sy = source_radius * np.cos(angle), source_radius * np.sin(angle)
        elif mode == "wave":
            displacement = 2.0 * float(amount) / 100.0
            sx = x + displacement * np.sin(np.pi * frequency * y + phase)
            sy = y + displacement * np.sin(np.pi * frequency * x + phase)
        elif mode == "radial_wave":
            source_radius = radius + 2.0 * float(amount) / 100.0 * np.sin(2.0 * np.pi * frequency * radius + phase)
            sx, sy = source_radius * np.cos(angle), source_radius * np.sin(angle)
        elif mode == "tunnel":
            zoom = math.exp(float(amount) / 100.0)
            source_radius = radius / max(1e-6, zoom)
            source_angle = angle + phase
            sx, sy = source_radius * np.cos(source_angle), source_radius * np.sin(source_angle)
        else:
            sector = 2.0 * np.pi / max(2, int(sides))
            source_angle = np.abs((angle + phase + sector * 0.5) % sector - sector * 0.5)
            source_angle += np.deg2rad(amount)
            sx, sy = radius * np.cos(source_angle), radius * np.sin(source_angle)
        grids.append(np.stack((sx, sy), axis=-1).astype(np.float32))
    return _motion_map(
        np.stack(grids, axis=0),
        fps=fps,
        operation=f"analytic:{mode}",
        parameters={
            "mode": mode,
            "amount": [amount_start, amount_end],
            "frequency": frequency,
            "phase_cycles": phase_cycles,
            "sides": int(sides),
            "easing": easing,
        },
    )


def displacement_motion_map(
    displacement: Any,
    frames: int,
    height: int,
    width: int,
    *,
    magnitude_percent: float = 10.0,
    encoding: str = "centered_rg",
    fps: float = 24.0,
) -> dict[str, Any]:
    """Import arbitrary vector data as a pullback displacement map.

    ``centered_rg`` interprets R/G in [0,1] with 0.5 as zero. ``signed_rg``
    interprets R/G directly as signed values.  A value of +1 moves the source
    lookup by ``magnitude_percent`` of the respective frame dimension.
    """

    frames, height, width = _dimensions(frames, height, width)
    if encoding not in {"centered_rg", "signed_rg"}:
        raise ValueError("displacement encoding must be centered_rg or signed_rg")
    source = np.asarray(displacement, dtype=np.float32)
    if source.ndim == 3:
        source = source[None, ...]
    if source.ndim != 4 or source.shape[-1] < 2:
        raise ValueError("displacement must have shape [frames,height,width,2+]")
    source = source[..., :2]
    if encoding == "centered_rg":
        source = 2.0 * source - 1.0
    target_grid = identity_grid(1, height, width)[0]
    positions = np.linspace(0.0, max(0, source.shape[0] - 1), frames, dtype=np.float32)
    fields = []
    for position in positions:
        low = int(math.floor(float(position)))
        high = min(source.shape[0] - 1, low + 1)
        fraction = float(position) - low
        first = _sample_numpy(source[low], target_grid, "border")
        if high == low:
            field = first
        else:
            second = _sample_numpy(source[high], target_grid, "border")
            field = first * (1.0 - fraction) + second * fraction
        fields.append(field.astype(np.float32))
    field = np.stack(fields)
    grid = identity_grid(frames, height, width) + 2.0 * float(magnitude_percent) * field / 100.0
    return _motion_map(
        grid,
        fps=fps,
        operation="displacement_import",
        parameters={
            "encoding": encoding,
            "magnitude_percent": magnitude_percent,
            "source_shape": list(np.asarray(displacement).shape),
        },
    )


def modulate_motion_map(
    value: dict[str, Any],
    *,
    strength_start: float = 0.0,
    strength_end: float = 1.0,
    easing: str = "smoothstep",
    spatial_mask: Any | None = None,
) -> dict[str, Any]:
    """Scale a map's displacement from identity with a temporal/spatial field."""

    validate_motion_map(value)
    frames, height, width = (
        int(value["frames"]),
        int(value["height"]),
        int(value["width"]),
    )
    strength = _lerp(strength_start, strength_end, _progress(frames, easing)).reshape(
        frames, 1, 1
    )
    if spatial_mask is None:
        field = np.ones((frames, height, width), dtype=np.float32)
    else:
        mask = np.asarray(spatial_mask, dtype=np.float32)
        if mask.ndim == 2:
            mask = mask[None, ...]
        if mask.ndim != 3:
            raise ValueError("spatial mask must have shape [frames,height,width]")
        target = identity_grid(1, height, width)[0]
        positions = np.linspace(0.0, max(0, mask.shape[0] - 1), frames)
        resized = []
        for position in positions:
            low = int(math.floor(float(position)))
            high = min(mask.shape[0] - 1, low + 1)
            fraction = float(position) - low
            first = _sample_numpy(mask[low], target, "border")[..., 0]
            if high == low:
                current = first
            else:
                second = _sample_numpy(mask[high], target, "border")[..., 0]
                current = first * (1.0 - fraction) + second * fraction
            resized.append(current)
        field = np.clip(np.stack(resized), 0.0, 1.0)
    amount = strength * field
    identity = identity_grid(frames, height, width)
    grid = identity + amount[..., None] * (np.asarray(value["grid"]) - identity)
    validity = validity_from_grid(grid) * (
        1.0 - np.clip(np.abs(amount), 0.0, 1.0)
        + np.clip(np.abs(amount), 0.0, 1.0) * np.asarray(value["validity"])
    )
    return _motion_map(
        grid,
        validity,
        fps=float(value["fps"]),
        time_domain=value["time_domain"],
        operation="modulate",
        parameters={
            "source_hash": value["tensor_hash"],
            "strength": [strength_start, strength_end],
            "easing": easing,
            "spatial_mask": spatial_mask is not None,
        },
        provenance=list(value.get("provenance", []))
        + [{"operation": value["operation"], "tensor_hash": value["tensor_hash"]}],
    )


def vector_field(
    frames: int,
    height: int,
    width: int,
    *,
    kind: str,
    duration_seconds: float,
    speed_x_percent: float = 0.0,
    speed_y_percent: float = 0.0,
    strength: float = 30.0,
    spatial_scale: float = 2.0,
    temporal_cycles: float = 0.0,
    temporal_mode: str = "forward",
    fps: float = 24.0,
) -> dict[str, Any]:
    if kind not in VECTOR_FIELDS:
        raise ValueError(f"vector field must be one of {', '.join(VECTOR_FIELDS)}")
    frames, height, width = _dimensions(frames, height, width)
    if float(duration_seconds) <= 0:
        raise ValueError("vector-field duration must be positive")
    if temporal_mode not in {"forward", "sine_loop"}:
        raise ValueError("temporal_mode must be forward or sine_loop")
    base = identity_grid(1, height, width)[0]
    x, y = base[..., 0], base[..., 1]
    radius2 = x * x + y * y
    times = np.linspace(0.0, float(duration_seconds), frames, dtype=np.float32)
    fields = []
    for time_index, time_value in enumerate(times):
        phase = 2.0 * np.pi * temporal_cycles * time_value / float(duration_seconds)
        loop_scale = math.pi * math.sin(2.0 * math.pi * time_index / max(1, frames - 1)) if temporal_mode == "sine_loop" else 1.0
        if kind == "uniform":
            vx = np.full_like(x, 2.0 * speed_x_percent / 100.0)
            vy = np.full_like(y, 2.0 * speed_y_percent / 100.0)
        elif kind == "rotation":
            omega = math.radians(strength)
            vx, vy = -omega * y, omega * x
        elif kind == "radial":
            rate = strength / 100.0
            vx, vy = rate * x, rate * y
        elif kind == "vortex":
            omega = math.radians(strength) * np.exp(-radius2 * max(0.01, spatial_scale))
            vx, vy = -omega * y, omega * x
        elif kind == "curl_sine":
            wave = max(0.01, spatial_scale) * np.pi
            amplitude = 2.0 * strength / 100.0
            vx = amplitude * wave * np.sin(wave * x + phase) * np.cos(wave * y + phase)
            vy = -amplitude * wave * np.cos(wave * x + phase) * np.sin(wave * y + phase)
        else:
            amplitude = 2.0 * strength / 100.0
            wave = max(0.01, spatial_scale) * np.pi
            vx = amplitude * np.sin(wave * y + phase)
            vy = amplitude * np.cos(wave * x + phase)
        fields.append(np.stack((vx * loop_scale, vy * loop_scale), axis=-1).astype(np.float32))
    values = np.stack(fields, axis=0)
    metadata = {
        "schema": VECTOR_FIELD_SCHEMA,
        "coordinate_system": COORDINATE_SYSTEM,
        "frames": frames,
        "height": height,
        "width": width,
        "fps": float(fps),
        "duration_seconds": float(duration_seconds),
        "units": "normalized_coordinates_per_second",
        "kind": kind,
        "temporal_mode": temporal_mode,
        "parameters": {
            "speed_percent": [speed_x_percent, speed_y_percent],
            "strength": strength,
            "spatial_scale": spatial_scale,
            "temporal_cycles": temporal_cycles,
        },
        "values": values,
        "tensor_hash": _array_hash(values),
    }
    return metadata


def validate_vector_field(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != VECTOR_FIELD_SCHEMA:
        raise ValueError(f"vector field schema must be {VECTOR_FIELD_SCHEMA}")
    values = np.asarray(value.get("values"), dtype=np.float32)
    expected = (int(value["frames"]), int(value["height"]), int(value["width"]), 2)
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError("vector-field tensor geometry is invalid")
    return value


def _sample_field(values: np.ndarray, time_position: float, coordinates: np.ndarray) -> np.ndarray:
    time_position = float(np.clip(time_position, 0.0, values.shape[0] - 1.0))
    low = int(math.floor(time_position))
    high = min(values.shape[0] - 1, low + 1)
    fraction = time_position - low
    first = _sample_numpy(values[low], coordinates, "border")
    if high == low:
        return first
    second = _sample_numpy(values[high], coordinates, "border")
    return first * (1.0 - fraction) + second * fraction


def integrate_advection(value: dict[str, Any], *, method: str = "rk2") -> dict[str, Any]:
    validate_vector_field(value)
    if method not in INTEGRATORS:
        raise ValueError(f"integrator must be one of {', '.join(INTEGRATORS)}")
    values = np.asarray(value["values"], dtype=np.float32)
    frames, height, width = values.shape[:3]
    base = identity_grid(1, height, width)[0]
    dt = float(value["duration_seconds"]) / max(1, frames - 1)
    maps = []
    for endpoint in range(frames):
        coordinates = base.copy()
        for step in range(endpoint, 0, -1):
            h = -dt
            if method == "euler":
                coordinates += h * _sample_field(values, float(step), coordinates)
            elif method == "rk2":
                k1 = _sample_field(values, float(step), coordinates)
                k2 = _sample_field(values, step - 0.5, coordinates + 0.5 * h * k1)
                coordinates += h * k2
            else:
                k1 = _sample_field(values, float(step), coordinates)
                k2 = _sample_field(values, step - 0.5, coordinates + 0.5 * h * k1)
                k3 = _sample_field(values, step - 0.5, coordinates + 0.5 * h * k2)
                k4 = _sample_field(values, float(step - 1), coordinates + h * k3)
                coordinates += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        maps.append(coordinates.astype(np.float32))
    provenance = [{"operation": "vector_field", "tensor_hash": value["tensor_hash"]}]
    return _motion_map(
        np.stack(maps, axis=0),
        fps=float(value["fps"]),
        operation="advection",
        parameters={"integrator": method, "field": value["kind"], "temporal_mode": value["temporal_mode"]},
        provenance=provenance,
    )


def _resize_map_arrays(
    value: dict[str, Any],
    frames: int,
    height: int,
    width: int,
    *,
    positions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    validate_motion_map(value)
    frames, height, width = _dimensions(frames, height, width)
    source_grid = np.asarray(value["grid"], dtype=np.float32)
    source_validity = np.asarray(value["validity"], dtype=np.float32)
    if positions is None:
        positions = np.linspace(0.0, 1.0, frames, dtype=np.float32)
    else:
        positions = np.asarray(positions, dtype=np.float32)
        if positions.shape != (frames,):
            raise ValueError("temporal resampling positions must match target frames")
    time_positions = np.clip(positions, 0.0, 1.0) * max(0, source_grid.shape[0] - 1)
    target_identity = identity_grid(1, height, width)[0]
    grids, validities = [], []
    for position in time_positions:
        low = int(math.floor(float(position)))
        high = min(source_grid.shape[0] - 1, low + 1)
        fraction = float(position) - low
        grid_low = _sample_numpy(source_grid[low], target_identity, "border")
        valid_low = _sample_numpy(source_validity[low], target_identity, "zeros")[..., 0]
        if high == low:
            grid, validity = grid_low, valid_low
        else:
            grid_high = _sample_numpy(source_grid[high], target_identity, "border")
            valid_high = _sample_numpy(source_validity[high], target_identity, "zeros")[..., 0]
            grid = grid_low * (1.0 - fraction) + grid_high * fraction
            validity = valid_low * (1.0 - fraction) + valid_high * fraction
        grids.append(grid.astype(np.float32))
        validities.append(np.clip(validity, 0.0, 1.0).astype(np.float32))
    return np.stack(grids), np.stack(validities)


def resample_motion_map(
    value: dict[str, Any],
    frames: int,
    height: int,
    width: int,
    *,
    positions: np.ndarray | None = None,
) -> dict[str, Any]:
    grid, validity = _resize_map_arrays(value, frames, height, width, positions=positions)
    return _motion_map(
        grid,
        validity,
        fps=float(value["fps"]),
        time_domain=value["time_domain"],
        operation=f"resample:{value['operation']}",
        parameters={"source_hash": value["tensor_hash"]},
        provenance=list(value.get("provenance", [])) + [{"operation": value["operation"], "tensor_hash": value["tensor_hash"]}],
    )


def compose_motion_maps(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Compose two pullbacks so media is sampled only once.

    The result is ``first(second(x))`` and matches applying ``first`` to the
    source, then applying ``second`` to that intermediate result.
    """

    validate_motion_map(first)
    validate_motion_map(second)
    frames = max(int(first["frames"]), int(second["frames"]))
    height, width = int(second["height"]), int(second["width"])
    first_grid, first_valid = _resize_map_arrays(first, frames, height, width)
    second_grid, second_valid = _resize_map_arrays(second, frames, height, width)
    grids, validities = [], []
    for index in range(frames):
        grid = _sample_numpy(first_grid[index], second_grid[index], "border")
        upstream = _sample_numpy(first_valid[index], second_grid[index], "zeros")[..., 0]
        grids.append(grid.astype(np.float32))
        validities.append(np.clip(second_valid[index] * upstream, 0.0, 1.0).astype(np.float32))
    provenance = list(first.get("provenance", [])) + [
        {"operation": first["operation"], "tensor_hash": first["tensor_hash"]},
        {"operation": second["operation"], "tensor_hash": second["tensor_hash"]},
    ]
    return _motion_map(
        np.stack(grids),
        np.stack(validities),
        fps=float(second["fps"]),
        time_domain=second["time_domain"],
        operation="compose",
        parameters={"first": first["tensor_hash"], "second": second["tensor_hash"]},
        provenance=provenance,
    )


def _rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    yaw, pitch, roll = map(math.radians, (yaw, pitch, roll))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return (rz @ rx @ ry).astype(np.float64)


def _resize_scalar(source: np.ndarray, height: int, width: int) -> np.ndarray:
    source = np.asarray(source, dtype=np.float32)
    if source.ndim == 3:
        source = source[..., 0]
    if source.ndim != 2:
        raise ValueError("depth must be a two-dimensional scalar field")
    target = identity_grid(1, height, width)[0]
    return _sample_numpy(source, target, "border")[..., 0].astype(np.float32)


def depth_camera_motion_map(
    depth: Any,
    frames: int,
    height: int,
    width: int,
    *,
    fov_degrees: float = 50.0,
    near: float = 1.0,
    far: float = 10.0,
    depth_mode: str = "near_white",
    translate_x_end: float = 0.0,
    translate_y_end: float = 0.0,
    translate_z_end: float = 20.0,
    yaw_end: float = 0.0,
    pitch_end: float = 0.0,
    roll_end: float = 0.0,
    easing: str = "smoothstep",
    fps: float = 24.0,
) -> dict[str, Any]:
    frames, height, width = _dimensions(frames, height, width)
    if not 1.0 <= float(fov_degrees) < 179.0:
        raise ValueError("field of view must lie in [1,179) degrees")
    if not 0.0 < float(near) < float(far):
        raise ValueError("depth planes require 0 < near < far")
    if depth_mode not in {"near_white", "near_black"}:
        raise ValueError("depth_mode must be near_white or near_black")
    depth = np.clip(_resize_scalar(np.asarray(depth), height, width), 0.0, 1.0)
    z = far - depth * (far - near) if depth_mode == "near_white" else near + depth * (far - near)
    base = identity_grid(1, height, width)[0]
    aspect = width / height
    tangent = math.tan(math.radians(float(fov_degrees)) * 0.5)
    world = np.stack(
        (base[..., 0] * z * tangent * aspect, -base[..., 1] * z * tangent, z),
        axis=-1,
    ).reshape(-1, 3)
    source_uv = base.reshape(-1, 2)
    median_depth = float(np.median(z))
    progress = _progress(frames, easing)
    grids, validities = [], []
    identity = base.copy()
    for fraction in progress:
        camera = median_depth * np.array(
            [translate_x_end, translate_y_end, translate_z_end], dtype=np.float64
        ) * float(fraction) / 100.0
        rotation = _rotation_matrix(
            yaw_end * float(fraction), pitch_end * float(fraction), roll_end * float(fraction)
        )
        camera_points = (world - camera) @ rotation
        positive = camera_points[:, 2] > 1e-6
        projected_x = camera_points[:, 0] / np.maximum(camera_points[:, 2], 1e-6) / (tangent * aspect)
        projected_y = -camera_points[:, 1] / np.maximum(camera_points[:, 2], 1e-6) / tangent
        pixel_x = (projected_x + 1.0) * width * 0.5 - 0.5
        pixel_y = (projected_y + 1.0) * height * 0.5 - 0.5
        candidates = []
        for x_index, x_weight in ((np.floor(pixel_x), 1.0 - (pixel_x - np.floor(pixel_x))), (np.floor(pixel_x) + 1.0, pixel_x - np.floor(pixel_x))):
            for y_index, y_weight in ((np.floor(pixel_y), 1.0 - (pixel_y - np.floor(pixel_y))), (np.floor(pixel_y) + 1.0, pixel_y - np.floor(pixel_y))):
                ix, iy = x_index.astype(np.int64), y_index.astype(np.int64)
                weight = x_weight * y_weight
                valid = positive & (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height) & (weight > 1e-6)
                candidates.append((iy[valid] * width + ix[valid], camera_points[valid, 2], -weight[valid], source_uv[valid], weight[valid]))
        flat = np.concatenate([item[0] for item in candidates])
        depths = np.concatenate([item[1] for item in candidates])
        negative_weights = np.concatenate([item[2] for item in candidates])
        uvs = np.concatenate([item[3] for item in candidates])
        weights = np.concatenate([item[4] for item in candidates])
        result = identity.reshape(-1, 2).copy()
        confidence = np.zeros(height * width, dtype=np.float32)
        if flat.size:
            order = np.lexsort((negative_weights, depths, flat))
            ordered_flat = flat[order]
            first = np.concatenate(([True], ordered_flat[1:] != ordered_flat[:-1]))
            selected = order[first]
            result[flat[selected]] = uvs[selected]
            confidence[flat[selected]] = weights[selected].astype(np.float32)
        grids.append(result.reshape(height, width, 2).astype(np.float32))
        validities.append(confidence.reshape(height, width))
    return _motion_map(
        np.stack(grids),
        np.stack(validities),
        fps=fps,
        operation="depth_camera_reprojection",
        parameters={
            "fov_degrees": fov_degrees,
            "near": near,
            "far": far,
            "depth_mode": depth_mode,
            "translation_end_percent_of_median_depth": [translate_x_end, translate_y_end, translate_z_end],
            "rotation_end_degrees": [yaw_end, pitch_end, roll_end],
            "easing": easing,
        },
    )


def _torch_grid(value: dict[str, Any], frames: int, height: int, width: int, device: Any, dtype: Any, positions: np.ndarray | None = None):
    import torch

    grid, validity = _resize_map_arrays(value, frames, height, width, positions=positions)
    return (
        torch.from_numpy(grid).to(device=device, dtype=dtype),
        torch.from_numpy(validity).to(device=device, dtype=torch.float32),
    )


def warp_images(images: Any, value: dict[str, Any], *, padding_mode: str = "border"):
    import torch
    import torch.nn.functional as functional

    validate_motion_map(value)
    if padding_mode not in PADDING_MODES:
        raise ValueError(f"padding mode must be one of {', '.join(PADDING_MODES)}")
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("Comfy IMAGE input must have shape [frames,height,width,channels]")
    source_frames, height, width, channels = map(int, images.shape)
    target_frames = int(value["frames"]) if source_frames == 1 else source_frames
    grid, validity = _torch_grid(value, target_frames, height, width, images.device, images.dtype)
    source = images.permute(0, 3, 1, 2)
    if source_frames == 1 and target_frames > 1:
        source = source.expand(target_frames, -1, -1, -1)
    elif source_frames != target_frames:
        raise ValueError("image batch must contain one frame or match the motion-map length")
    warped = functional.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=False,
    ).permute(0, 2, 3, 1)
    return warped, validity
