"""Minimal native plate-sketch image operations."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


def parse_rgb(value: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(value).strip())
    if not match:
        raise ValueError("colour must be a six-digit hex value, for example #10190B")
    raw = match.group(1)
    return tuple(int(raw[index : index + 2], 16) for index in (0, 2, 4))


def create_canvas(width: int, height: int, background: str):
    import torch

    if int(width) < 1 or int(height) < 1:
        raise ValueError("canvas dimensions must be positive")
    colour = torch.tensor(parse_rgb(background), dtype=torch.float32).div_(255.0)
    return colour.view(1, 1, 1, 3).expand(1, int(height), int(width), 3).clone()


def composite_layer(
    canvas: Any,
    layer: Any,
    *,
    x_percent: float = 50.0,
    y_percent: float = 50.0,
    scale: float = 1.0,
    rotation: float = 0.0,
    opacity: float = 1.0,
    blend_mode: str = "normal",
    feather_pixels: int = 0,
    mask: Any = None,
):
    import numpy as np
    import torch
    from PIL import Image, ImageFilter

    if blend_mode not in {"normal", "screen", "multiply", "add"}:
        raise ValueError("unsupported plate blend mode")
    if float(scale) <= 0:
        raise ValueError("layer scale must be positive")
    if not 0.0 <= float(opacity) <= 1.0:
        raise ValueError("opacity must be between 0 and 1")

    base = np.clip(canvas[0].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    source = np.clip(layer[0].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    base_h, base_w = base.shape[:2]
    source_image = Image.fromarray(source[..., :3], mode="RGB")
    target_size = (
        max(1, int(round(source_image.width * float(scale)))),
        max(1, int(round(source_image.height * float(scale)))),
    )
    source_image = source_image.resize(target_size, Image.Resampling.LANCZOS)
    source_image = source_image.rotate(
        -float(rotation), resample=Image.Resampling.BICUBIC, expand=True
    )

    if mask is None:
        alpha_image = Image.new("L", source_image.size, 255)
    else:
        raw_mask = np.clip(mask[0].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        alpha_image = Image.fromarray(raw_mask, mode="L").resize(
            target_size, Image.Resampling.BILINEAR
        )
        alpha_image = alpha_image.rotate(
            -float(rotation), resample=Image.Resampling.BILINEAR, expand=True
        )
    alpha_image = alpha_image.point(lambda value: int(value * float(opacity)))

    feather = max(0, int(feather_pixels))
    if feather:
        pad = feather * 2
        padded_size = (source_image.width + pad * 2, source_image.height + pad * 2)
        padded_source = Image.new("RGB", padded_size)
        padded_source.paste(source_image, (pad, pad))
        padded_alpha = Image.new("L", padded_size, 0)
        padded_alpha.paste(alpha_image, (pad, pad))
        source_image = padded_source
        alpha_image = padded_alpha.filter(ImageFilter.GaussianBlur(radius=feather))

    source_array = np.asarray(source_image, dtype=np.float32).copy()
    alpha = np.asarray(alpha_image, dtype=np.float32).copy() / 255.0
    left = int(round(float(x_percent) / 100.0 * base_w - source_array.shape[1] / 2))
    top = int(round(float(y_percent) / 100.0 * base_h - source_array.shape[0] / 2))
    right, bottom = left + source_array.shape[1], top + source_array.shape[0]
    x0, y0, x1, y1 = max(0, left), max(0, top), min(base_w, right), min(base_h, bottom)
    full_mask = np.zeros((base_h, base_w), dtype=np.float32)
    if x1 > x0 and y1 > y0:
        sx0, sy0 = x0 - left, y0 - top
        sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
        foreground = source_array[sy0:sy1, sx0:sx1]
        local_alpha = alpha[sy0:sy1, sx0:sx1][..., None]
        background = base[y0:y1, x0:x1].astype(np.float32)
        if blend_mode == "multiply":
            blended = background * foreground / 255.0
        elif blend_mode == "screen":
            blended = 255.0 - ((255.0 - background) * (255.0 - foreground) / 255.0)
        elif blend_mode == "add":
            blended = np.clip(background + foreground, 0.0, 255.0)
        else:
            blended = foreground
        base[y0:y1, x0:x1] = np.clip(
            background * (1.0 - local_alpha) + blended * local_alpha, 0, 255
        ).astype(np.uint8)
        full_mask[y0:y1, x0:x1] = alpha[sy0:sy1, sx0:sx1]

    composite = torch.from_numpy(base.astype(np.float32) / 255.0).unsqueeze(0)
    layer_mask = torch.from_numpy(full_mask).unsqueeze(0)
    return (
        composite.to(device=canvas.device, dtype=canvas.dtype),
        layer_mask.to(device=canvas.device),
    )


def domemaster_preview(image: Any, outside_level: float = 0.0, edge_feather_pixels: int = 12):
    import torch

    height, width = int(image.shape[1]), int(image.shape[2])
    yy = torch.arange(height, device=image.device, dtype=torch.float32).view(height, 1)
    xx = torch.arange(width, device=image.device, dtype=torch.float32).view(1, width)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    radius = min(width, height) / 2.0
    distance = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    feather = max(0.0, float(edge_feather_pixels))
    mask = (
        ((radius - distance) / feather).clamp(0.0, 1.0)
        if feather
        else (distance <= radius).to(torch.float32)
    )
    level = min(1.0, max(0.0, float(outside_level)))
    preview = image * mask.view(1, height, width, 1) + level * (
        1.0 - mask.view(1, height, width, 1)
    )
    return preview, mask.unsqueeze(0)


def write_plate_sidecars(
    png_path: str | Path,
    *,
    point: dict[str, Any],
    prompt: str,
    extra: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    png = Path(png_path).resolve()
    prompt_path = png.with_suffix(".prompt.txt")
    manifest_path = png.with_suffix(".json")
    prompt_path.write_text(str(prompt).rstrip() + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "cauce.plate-handoff/1",
                "png": png.name,
                "point": point,
                "prompt": str(prompt),
                "extra": extra or {},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return prompt_path, manifest_path
