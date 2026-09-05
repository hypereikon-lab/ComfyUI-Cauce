"""Experimental frequency-selective H3 RoPE using calibrated Zenith-180 rays.

This changes phases, not position IDs or latent samples. It is an inference
ablation, not a learned lens model or a guarantee of spherical content.
"""
from __future__ import annotations

import logging
import math

PROFILE = "zenith-180-hybrid-rope/1"
WRAPPER_KEY = "cauce_h3_zenith_rope"
LOG = logging.getLogger(__name__)


def zenith_ray(u: float, v: float):
    """Square, full short-edge circle; +Y zenith, +X right, +Z image-up."""
    x, y = 2 * float(u) - 1, 1 - 2 * float(v)
    radius = math.hypot(x, y)
    if radius > 1 + 1e-12:
        return None
    if radius < 1e-12:
        return (0.0, 1.0, 0.0)
    theta = min(radius, 1.0) * math.pi / 2
    s = math.sin(theta) / radius
    return (x * s, math.cos(theta), y * s)


def validate_options(strength, low_frequency_count):
    if not math.isfinite(float(strength)) or not 0 <= float(strength) <= 1:
        raise ValueError("strength must be finite and in [0,1]")
    if int(low_frequency_count) != low_frequency_count or not 1 <= low_frequency_count <= 15:
        raise ValueError("low_frequency_count must be in [1,15], retaining local bands")


def hybrid_phases(stock, position_ids, segments, inv_freq, latent_height,
                  latent_width, strength, low_frequency_count, include_keyframes=True):
    """Return duplicated [t,h,w,t,h,w] phases with only selected rows/bands changed.

    Lowest actual |inv_freq| select the band; high bands, time, other modalities
    and exterior rows are copied exactly. Center calibration matches the first
    derivative of stock h/w. Paired tilts include the zenith ray component.
    """
    import torch

    validate_options(strength, low_frequency_count)
    if strength == 0:
        return stock, {"profile": PROFILE, "bypass": True}
    if latent_height != latent_width or latent_height % 2 or latent_height < 4:
        raise ValueError("Zenith profile requires square, even spatial latents")
    if inv_freq.numel() != 16 or stock.shape != (position_ids.shape[0], 96):
        raise ValueError("Unsupported H3 RoPE layout: expected 16 frequencies per axis")
    n = latent_height // 2
    frame_rows = n * n
    inv = inv_freq.to(device=stock.device, dtype=stock.dtype).flatten()
    selected = torch.argsort(inv.abs())[:low_frequency_count].sort().values
    coord = (torch.arange(n, device=stock.device, dtype=stock.dtype) + .5) * (2 / n) - 1
    yy, xx = torch.meshgrid(-coord, coord, indexing="ij")
    r = torch.sqrt(xx.square() + yy.square())
    valid = r <= 1
    theta = r.clamp(max=1) * (math.pi / 2)
    factor = torch.sin(theta) / r.clamp(min=1e-12)
    dx, dy, dz = xx * factor, torch.cos(theta), yy * factor
    # h points down (-Z), w points right (+X); +/-Y tilts add curvature
    # without replacing the nonzero local tangent derivatives by a radial-only axis.
    signs = torch.where(torch.arange(low_frequency_count, device=stock.device) % 2 == 0, 1., -1.)
    hgeo = (-dz.flatten()[:, None] + (dy.flatten()[:, None] - 1) * signs) * (16 / (math.pi / 2))
    wgeo = (dx.flatten()[:, None] - (dy.flatten()[:, None] - 1) * signs) * (16 / (math.pi / 2))
    result = stock.clone()
    report_segments = []
    maxima = []
    for start, stop, kind in segments:
        if kind not in ({"video", "cond"} if include_keyframes else {"video"}):
            continue
        count = stop - start
        if count <= 0 or count % frame_rows:
            raise ValueError("Selected H3 segment does not share the target spatial grid")
        # Check the coordinate contract instead of assuming all conditions share it.
        expected_axis = torch.arange(n, dtype=position_ids.dtype, device=position_ids.device) * (32 / n)
        eh, ew = torch.meshgrid(expected_axis, expected_axis, indexing="ij")
        expected = torch.stack((eh.flatten(), ew.flatten()), dim=1).repeat(count // frame_rows, 1)
        if not torch.allclose(position_ids[start:stop, 1:], expected, atol=1e-6, rtol=1e-6):
            raise ValueError("Selected segment has nonstandard spatial positions; refusing to stack coordinate patches")
        local_rows = torch.nonzero(valid.flatten(), as_tuple=False).flatten()
        rows = (torch.arange(count // frame_rows, device=stock.device)[:, None] * frame_rows + local_rows).flatten() + start
        for offset, geometry in ((16, hgeo), (32, wgeo)):
            # Stock uses left-edge IDs; retain its exact mean phase origin.
            center = 16 * (n - 1) / n
            target = ((center + geometry[local_rows]) * inv[selected]).repeat(count // frame_rows, 1)
            for duplicate in (0, 48):
                columns = selected + offset + duplicate
                original = stock[rows[:, None], columns]
                delta = (target - original) * strength
                result[rows[:, None], columns] = original + delta
                maxima.append(delta.abs().max())
        report_segments.append({"kind": kind, "rows": count, "interior_rows": rows.numel()})
    if not any(item['kind'] == 'video' for item in report_segments):
        raise ValueError("No target-video segment found")
    return result, {
        "profile": PROFILE, "bypass": False, "strength": strength,
        "low_frequency_indices": selected.tolist(),
        "selected_inv_freq": inv[selected].tolist(),
        "max_phase_delta_radians": torch.stack(maxima).max().item(),
        "spatial_token_grid": [n, n], "segments": report_segments,
        "time_unchanged": True, "high_bands_unchanged": True,
        "exterior_unchanged": True,
    }


class H3ZenithRoPEPatch:
    def __init__(self, strength, low_frequency_count, include_keyframes=True):
        validate_options(strength, low_frequency_count)
        self.strength = float(strength)
        self.low_frequency_count = int(low_frequency_count)
        self.include_keyframes = bool(include_keyframes)
        self.last_report = None

    def __call__(self, executor, x, timestep, context, transformer_options=None, **kwargs):
        if self.strength == 0:
            return executor(x, timestep, context, transformer_options or {}, **kwargs)
        model = executor.class_obj
        layout = (kwargs.get("minimax_payload") or {}).get("layout")
        if layout is None or not hasattr(model, 'rope_freqs') or not hasattr(model, 'rope'):
            raise ValueError("Zenith RoPE requires native H3 PackedLayout and rope_freqs")
        original = model.rope_freqs
        had_instance_override = 'rope_freqs' in vars(model)
        saved_override = vars(model).get('rope_freqs')
        called = False

        def altered(position_ids, device):
            nonlocal called
            called = True
            stock = original(position_ids, device)
            result, report = hybrid_phases(
                stock, position_ids, layout.segments, model.rope.inv_freq,
                x[0].shape[-2], x[0].shape[-1], self.strength,
                self.low_frequency_count, self.include_keyframes,
            )
            if self.last_report is None:
                import json
                LOG.info("CAUCE_ZENITH_ROPE %s", json.dumps(report, sort_keys=True))
            self.last_report = report
            return result

        # A per-instance override exists only during this synchronous forward.
        # Comfy's normal executor is serial. Concurrent forwards/torch.compile
        # on the same shared diffusion model are outside this experiment's contract.
        model.rope_freqs = altered
        try:
            output = executor(x, timestep, context, transformer_options or {}, **kwargs)
            if not called:
                raise RuntimeError("H3 did not consume the Zenith phase hook")
            return output
        finally:
            if had_instance_override:
                model.rope_freqs = saved_override
            else:
                delattr(model, 'rope_freqs')
