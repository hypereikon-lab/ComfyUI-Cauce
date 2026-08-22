"""Versioned, media-first contracts used by CAUCE nodes and the runner."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Literal

from .timebase import (
    H3_FPS,
    H3Shape,
    as_fraction,
    format_seconds,
    format_timecode,
    fraction_from_payload,
    fraction_payload,
    frames_to_seconds,
    is_h3_frame_count,
    round_fraction,
    snap_h3_frame_count,
    visual_token_count_for_span,
)


PROJECT_SCHEMA = "cauce.project/1"
TIMELINE_SCHEMA = "cauce.timeline/1"
POINT_SCHEMA = "cauce.point/1"
SPAN_SCHEMA = "cauce.span/1"
FIELD_SCHEMA = "cauce.field/1"
WINDOW_SCHEMA = "cauce.window/1"
DECODE_DOMAIN_SCHEMA = "cauce.decode-domain/1"
RECEIPT_SCHEMA = "cauce.receipt/1"
PROFILE_SCHEMA = "cauce.execution-profile/1"
SEAM_SCHEMA = "cauce.seam/1"

MEDIA_KINDS = ("image", "video", "audio", "mask", "latent")
FIELD_CHANNELS = ("video", "audio", "both")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_id(value: object, fallback: str = "untitled") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("-.")
    return cleaned or fallback


def _time(value: object) -> dict[str, int]:
    return fraction_payload(fraction_from_payload(value))


def _range_payload(start: object, end: object) -> dict[str, object]:
    start_f = fraction_from_payload(start)
    end_f = fraction_from_payload(end)
    if start_f < 0:
        raise ValueError("time ranges cannot start before zero")
    if end_f <= start_f:
        raise ValueError("time ranges require end > start")
    return {
        "start": fraction_payload(start_f),
        "end": fraction_payload(end_f),
        "duration": fraction_payload(end_f - start_f),
        "start_seconds": float(start_f),
        "end_seconds": float(end_f),
        "duration_seconds": float(end_f - start_f),
    }


def range_fraction(value: dict[str, Any]) -> tuple[Any, Any]:
    return fraction_from_payload(value["start"]), fraction_from_payload(value["end"])


def make_asset(
    asset_id: str,
    kind: Literal["image", "video", "audio", "mask", "latent"],
    source: str = "",
    *,
    sha256: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if kind not in MEDIA_KINDS:
        raise ValueError(f"unsupported media kind: {kind}")
    return {
        "schema": "cauce.asset/1",
        "id": safe_id(asset_id, "asset"),
        "kind": kind,
        "source": str(source),
        "sha256": str(sha256),
        "metadata": copy.deepcopy(metadata or {}),
    }


def make_point(
    point_id: str,
    master_time: object,
    prompt: str = "",
    *,
    assets: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    time_f = fraction_from_payload(master_time)
    if time_f < 0:
        raise ValueError("point time cannot be negative")
    point = {
        "schema": POINT_SCHEMA,
        "id": safe_id(point_id, "point"),
        "time": fraction_payload(time_f),
        "seconds": float(time_f),
        "timecode": format_timecode(time_f),
        "prompt": str(prompt),
        "assets": [copy.deepcopy(asset) for asset in assets],
    }
    point["hash"] = content_hash({key: value for key, value in point.items() if key != "hash"})
    return point


def make_span(
    span_id: str,
    kind: str,
    start: object,
    end: object,
    *,
    source: str = "",
    offset: object = 0,
    asset_id: str = "",
) -> dict[str, Any]:
    if kind not in MEDIA_KINDS:
        raise ValueError(f"unsupported media kind: {kind}")
    offset_f = fraction_from_payload(offset)
    if offset_f < 0:
        raise ValueError("media offset cannot be negative")
    return {
        "schema": SPAN_SCHEMA,
        "id": safe_id(span_id, "span"),
        "kind": kind,
        "asset_id": safe_id(asset_id, "asset") if asset_id else "",
        "source": str(source),
        "range": _range_payload(start, end),
        "offset": fraction_payload(offset_f),
        "offset_seconds": float(offset_f),
    }


def make_field() -> dict[str, Any]:
    return {"schema": FIELD_SCHEMA, "spans": []}


def append_field_span(
    field: dict[str, Any] | None,
    *,
    channel: Literal["video", "audio", "both"],
    start: object,
    end: object,
    strength: float,
) -> dict[str, Any]:
    if channel not in FIELD_CHANNELS:
        raise ValueError(f"unsupported field channel: {channel}")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("field strength must be between 0 and 1")
    out = copy.deepcopy(field or make_field())
    if out.get("schema") != FIELD_SCHEMA:
        raise ValueError(f"field schema must be {FIELD_SCHEMA}")
    out.setdefault("spans", []).append(
        {
            "channel": channel,
            "range": _range_payload(start, end),
            "strength": float(strength),
        }
    )
    out["spans"].sort(key=lambda item: fraction_from_payload(item["range"]["start"]))
    return out


def make_window(
    window_id: str,
    accepted_start: object,
    accepted_duration: object,
    *,
    context_frames: int = 0,
    duplicate_prefix_frames: int = 0,
    snap_mode: str = "ceil",
    accept_mode: str = "nearest_run",
    maximum_frames: int = 362,
) -> dict[str, Any]:
    accepted_start_f = fraction_from_payload(accepted_start)
    accepted_duration_f = fraction_from_payload(accepted_duration)
    if accepted_start_f < 0:
        raise ValueError("accepted_start cannot be negative")
    if accepted_duration_f <= 0:
        raise ValueError("accepted_duration must be positive")
    context_frames = int(context_frames)
    duplicate_prefix_frames = int(duplicate_prefix_frames)
    if context_frames < 0 or duplicate_prefix_frames < 0:
        raise ValueError("context and duplicate-prefix frames cannot be negative")
    if context_frames and not is_h3_frame_count(context_frames):
        raise ValueError(
            "visual continuation context must use the H3 grid: 5, 22, 39, 56, ... frames"
        )
    if duplicate_prefix_frames:
        visual_token_count_for_span(duplicate_prefix_frames)

    # Context and a decoder's repeated prefix are distinct facts, but they can
    # occupy the same head of a window.  They are therefore not blindly added:
    # accepting begins after whichever hidden region reaches farther.
    hidden_head_frames = max(context_frames, duplicate_prefix_frames)
    requested_hidden_head = frames_to_seconds(hidden_head_frames)
    actual_hidden_head = min(requested_hidden_head, accepted_start_f)
    if actual_hidden_head != requested_hidden_head and hidden_head_frames:
        # A shortened head would no longer land on the requested token boundary.
        raise ValueError(
            "the window starts before its requested context/prefix; reduce the head or move accepted_start"
        )
    render_start = accepted_start_f - actual_hidden_head
    accepted_offset_frames = hidden_head_frames
    requested_total = frames_to_seconds(accepted_offset_frames) + accepted_duration_f
    frames = snap_h3_frame_count(
        requested_total,
        unit="seconds",
        mode=snap_mode,  # type: ignore[arg-type]
        maximum=maximum_frames,
    )
    shape = H3Shape.from_frames(frames)
    render_end = render_start + shape.duration
    accepted_start_render = accepted_start_f
    requested_accepted_end = accepted_start_f + accepted_duration_f
    requested_accepted_frames = max(1, round_fraction(accepted_duration_f * H3_FPS))
    requested_end_frame = accepted_offset_frames + requested_accepted_frames
    run_boundaries = tuple(
        frame
        for frame in range(5, frames + 1)
        if frame > accepted_offset_frames and is_h3_frame_count(frame)
    )
    if accept_mode == "exact_frames":
        accepted_end_frame = requested_end_frame
    elif accept_mode == "floor_run":
        candidates = tuple(
            boundary
            for boundary in run_boundaries
            if boundary <= requested_end_frame
        )
        if not candidates:
            raise ValueError(
                "accepted duration does not reach the next phase-safe H3 run boundary"
            )
        accepted_end_frame = candidates[-1]
    elif accept_mode == "ceil_run":
        candidates = tuple(
            boundary
            for boundary in run_boundaries
            if boundary >= requested_end_frame
        )
        if not candidates:
            raise ValueError(
                "render window does not reach a phase-safe H3 run boundary after the requested end"
            )
        accepted_end_frame = candidates[0]
    elif accept_mode == "nearest_run":
        if not run_boundaries:
            raise ValueError("render window contains no phase-safe accepted H3 run")
        accepted_end_frame = min(
            run_boundaries,
            key=lambda boundary: (abs(boundary - requested_end_frame), boundary),
        )
    elif accept_mode == "full_render":
        accepted_end_frame = frames
    else:
        raise ValueError(
            "accept_mode must be nearest_run, floor_run, ceil_run, exact_frames, or full_render"
        )
    if accepted_end_frame > frames:
        raise ValueError(
            "accepted range extends past the rendered window; use ceil snapping or a larger window"
        )
    accepted_frames = accepted_end_frame - accepted_offset_frames
    accepted_end = render_start + frames_to_seconds(accepted_end_frame)
    if accepted_end <= accepted_start_render:
        raise ValueError("the generated window leaves no accepted media")

    window = {
        "schema": WINDOW_SCHEMA,
        "id": safe_id(window_id, "window"),
        "fps": fraction_payload(H3_FPS),
        "shape": shape.to_dict(),
        "context_frames": context_frames,
        "duplicate_prefix_frames": duplicate_prefix_frames,
        "accept_mode": accept_mode,
        "accepted_offset_frames": accepted_offset_frames,
        "accepted_start_frame": accepted_offset_frames,
        "accepted_end_frame": accepted_end_frame,
        "accepted_frames": accepted_frames,
        "phase_safe_parent": is_h3_frame_count(accepted_end_frame),
        "render_range": _range_payload(render_start, render_end),
        "accepted_range": _range_payload(accepted_start_render, accepted_end),
        "requested_accepted_range": _range_payload(
            accepted_start_f, requested_accepted_end
        ),
        "context_range": (
            _range_payload(render_start, render_start + frames_to_seconds(context_frames))
            if context_frames
            else None
        ),
        "duplicate_prefix_range": (
            _range_payload(
                render_start,
                render_start + frames_to_seconds(duplicate_prefix_frames),
            )
            if duplicate_prefix_frames
            else None
        ),
        "discard_after_range": (
            _range_payload(accepted_end, render_end) if render_end > accepted_end else None
        ),
        "timecode": format_timecode(render_start),
    }
    window["hash"] = content_hash({key: value for key, value in window.items() if key != "hash"})
    return window


def make_timeline(timeline_id: str = "main") -> dict[str, Any]:
    return {
        "schema": TIMELINE_SCHEMA,
        "id": safe_id(timeline_id, "main"),
        "points": [],
        "spans": [],
        "windows": [],
    }


def make_project(
    project_id: str,
    title: str,
    *,
    timeline: dict[str, Any] | None = None,
    output_root: str = "cauce",
) -> dict[str, Any]:
    project = {
        "schema": PROJECT_SCHEMA,
        "id": safe_id(project_id, "project"),
        "title": str(title),
        "clock": {"kind": "rational", "video_fps": fraction_payload(H3_FPS)},
        "output_root": str(output_root).strip() or "cauce",
        "timeline": copy.deepcopy(timeline or make_timeline()),
    }
    project["hash"] = content_hash({key: value for key, value in project.items() if key != "hash"})
    return project


def append_timeline_item(
    timeline: dict[str, Any] | None,
    item: dict[str, Any],
) -> dict[str, Any]:
    out = copy.deepcopy(timeline or make_timeline())
    if out.get("schema") != TIMELINE_SCHEMA:
        raise ValueError(f"timeline schema must be {TIMELINE_SCHEMA}")
    schema = item.get("schema")
    destination = {
        POINT_SCHEMA: "points",
        SPAN_SCHEMA: "spans",
        WINDOW_SCHEMA: "windows",
    }.get(schema)
    if destination is None:
        raise ValueError(f"cannot append schema {schema!r} to a CAUCE timeline")
    out[destination].append(copy.deepcopy(item))
    if destination == "points":
        out[destination].sort(key=lambda value: fraction_from_payload(value["time"]))
    else:
        out[destination].sort(
            key=lambda value: fraction_from_payload(
                value.get("range", value.get("render_range"))["start"]
            )
        )
    out["hash"] = content_hash({key: value for key, value in out.items() if key != "hash"})
    return out


def make_decode_domain(
    domain_id: str,
    start: object,
    end: object,
    *,
    artifact_ids: Iterable[str] = (),
) -> dict[str, Any]:
    return {
        "schema": DECODE_DOMAIN_SCHEMA,
        "id": safe_id(domain_id, "decode"),
        "range": _range_payload(start, end),
        "artifact_ids": [str(value) for value in artifact_ids],
    }


def make_receipt(
    artifact_id: str,
    *,
    parents: Iterable[str] = (),
    window: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
    seed: int | None = None,
    sampler: str = "",
    scheduler: str = "",
    steps: int | None = None,
    cfg: float | None = None,
    model_hashes: dict[str, str] | None = None,
    workflow_hash: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "artifact_id": safe_id(artifact_id, "artifact"),
        "parents": [str(value) for value in parents],
        "window": copy.deepcopy(window),
        "profile": copy.deepcopy(profile),
        "seed": int(seed) if seed is not None else None,
        "sampler": str(sampler),
        "scheduler": str(scheduler),
        "steps": int(steps) if steps is not None else None,
        "cfg": float(cfg) if cfg is not None else None,
        "model_hashes": dict(model_hashes or {}),
        "workflow_hash": str(workflow_hash),
        "extra": copy.deepcopy(extra or {}),
    }
    receipt["receipt_hash"] = content_hash(receipt)
    return receipt


def window_summary(window: dict[str, Any]) -> str:
    if window.get("schema") != WINDOW_SCHEMA:
        raise ValueError(f"window schema must be {WINDOW_SCHEMA}")
    render_start, render_end = range_fraction(window["render_range"])
    accept_start, accept_end = range_fraction(window["accepted_range"])
    shape = window["shape"]
    return (
        f"{window['id']}: render {format_seconds(render_start)}-{format_seconds(render_end)}s "
        f"({shape['pixel_frames']}f/{shape['video_latent_frames']}v/{shape['audio_latent_frames']}a), "
        f"accept {format_seconds(accept_start)}-{format_seconds(accept_end)}s"
    )
