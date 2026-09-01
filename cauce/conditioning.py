"""Read-only inspection of MiniMax H3 conditioning metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import H3_PACKED_SEQUENCE_REPORT_SCHEMA, content_hash
from .h3 import validate_av_latent
from .timebase import visual_span_for_tokens


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return [int(item) for item in shape] if shape is not None else None


def _guide_video(value: Any) -> Any:
    if isinstance(value, Mapping) and "samples" in value:
        return value["samples"]
    return value


def inspect_h3_conditioning(
    positive: Sequence[Any],
    target_av_latent: Mapping[str, Any],
    *,
    timeline_origin_frame: int = 0,
) -> dict[str, Any]:
    """Validate and serialize the H3-specific metadata on active conditioning."""

    if not isinstance(positive, (list, tuple)) or not positive:
        raise TypeError("positive must be a non-empty ComfyUI CONDITIONING sequence")
    _, _, target_frames = validate_av_latent(
        target_av_latent,
        timeline_origin_frame=int(timeline_origin_frame),
        name="target_av_latent",
    )
    entries: list[dict[str, Any]] = []
    all_ranges: list[dict[str, int]] = []
    total_keyframes = 0
    total_references = 0
    for entry_index, entry in enumerate(positive):
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise TypeError("conditioning entries must contain a tensor and metadata")
        metadata = entry[1]
        if not isinstance(metadata, Mapping):
            raise TypeError("conditioning metadata must be a mapping")
        keyframes = metadata.get("minimax_keyframes", [])
        references = metadata.get("minimax_refs", [])
        if not isinstance(keyframes, (list, tuple)):
            raise TypeError("minimax_keyframes must be a list or tuple")
        if not isinstance(references, (list, tuple)):
            raise TypeError("minimax_refs must be a list or tuple")

        keyframe_reports: list[dict[str, Any]] = []
        for keyframe_index, keyframe in enumerate(keyframes):
            if not isinstance(keyframe, Mapping):
                raise TypeError("every minimax_keyframes entry must be a mapping")
            resolved = keyframe.get("resolved_frame_index")
            if not isinstance(resolved, int):
                raise ValueError("every H3 keyframe must expose an integer resolved_frame_index")
            video = _guide_video(keyframe.get("latent"))
            video_shape = _shape(video)
            audio = keyframe.get("audio_latent")
            audio_shape = _shape(audio)
            if video_shape is None and audio_shape is None:
                raise ValueError("an H3 keyframe must contain a video or audio latent")
            guide_frames = 1
            if video_shape is not None:
                if len(video_shape) != 5 or video_shape[1] != 24:
                    raise ValueError("H3 guide video latent has invalid geometry")
                guide_frames = visual_span_for_tokens(video_shape[2])
            if audio_shape is not None and (
                len(audio_shape) != 4 or audio_shape[1:3] != [32, 2]
            ):
                raise ValueError("H3 guide audio latent has invalid geometry")
            end = resolved + guide_frames
            if resolved < 0 or resolved >= target_frames or end > target_frames:
                raise ValueError("H3 keyframe lies outside the target latent")
            report = {
                "index": keyframe_index,
                "resolved_frame_index": resolved,
                "guide_frames": guide_frames,
                "guide_range": [resolved, end],
                "video_shape": video_shape,
                "audio_shape": audio_shape,
            }
            keyframe_reports.append(report)
            all_ranges.append(
                {
                    "entry": entry_index,
                    "keyframe": keyframe_index,
                    "start": resolved,
                    "end": end,
                }
            )

        reference_reports: list[dict[str, Any]] = []
        for reference_index, reference in enumerate(references):
            if not isinstance(reference, Mapping):
                raise TypeError("every minimax_refs entry must be a mapping")
            reference_reports.append(
                {
                    "index": reference_index,
                    "kind": reference.get("kind"),
                    "latent_shape": _shape(reference.get("latent")),
                    "audio_latent_shape": _shape(reference.get("audio_latent")),
                    "latent_t": reference.get("latent_t"),
                    "ref_audio_t": reference.get("ref_audio_t"),
                }
            )
        total_keyframes += len(keyframe_reports)
        total_references += len(reference_reports)
        entries.append(
            {
                "index": entry_index,
                "keyframes": keyframe_reports,
                "references": reference_reports,
            }
        )

    overlaps: list[dict[str, Any]] = []
    for left_index, left in enumerate(all_ranges):
        for right in all_ranges[left_index + 1 :]:
            if left["entry"] != right["entry"]:
                continue
            start = max(left["start"], right["start"])
            end = min(left["end"], right["end"])
            if start < end:
                overlaps.append(
                    {
                        "entry": left["entry"],
                        "keyframes": [left["keyframe"], right["keyframe"]],
                        "range": [start, end],
                    }
                )

    payload: dict[str, Any] = {
        "schema": "cauce.h3-conditioning-report/1",
        "target_frames": target_frames,
        "timeline_origin_frame": int(timeline_origin_frame),
        "conditioning_entries": len(entries),
        "keyframe_count": total_keyframes,
        "reference_count": total_references,
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
        "entries": entries,
    }
    payload["report_hash"] = content_hash(payload)
    return payload


def _reference_geometry(reference: Mapping[str, Any]) -> tuple[int, int]:
    latent_h = reference.get("latent_h")
    latent_w = reference.get("latent_w")
    if isinstance(latent_h, int) and isinstance(latent_w, int):
        return latent_h, latent_w
    latent_shape = _shape(reference.get("latent"))
    if latent_shape is None or len(latent_shape) != 5:
        raise ValueError("visual H3 references must expose latent_h/latent_w or a 5D latent")
    return int(latent_shape[3]), int(latent_shape[4])


def inspect_h3_packed_sequence(
    positive: Sequence[Any],
    target_av_latent: Mapping[str, Any],
    *,
    timeline_origin_frame: int = 0,
    estimated_bytes_per_row: int = 151_000,
) -> dict[str, Any]:
    """Count H3 packed rows without importing or patching ComfyUI internals.

    The row count mirrors ``PackedLayout`` in official ComfyUI. The byte estimate
    is intentionally separate: it is a user-visible calibration knob, not a claim
    that memory use is a fixed linear function on every backend.
    """

    if not isinstance(positive, (list, tuple)) or not positive:
        raise TypeError("positive must be a non-empty ComfyUI CONDITIONING sequence")
    video, audio, target_frames = validate_av_latent(
        target_av_latent,
        timeline_origin_frame=int(timeline_origin_frame),
        name="target_av_latent",
    )
    calibration = int(estimated_bytes_per_row)
    if calibration < 1:
        raise ValueError("estimated_bytes_per_row must be positive")
    latent_t = int(video.shape[2])
    latent_h = ((int(video.shape[3]) + 1) // 2) * 2
    latent_w = ((int(video.shape[4]) + 1) // 2) * 2
    audio_t = int(audio.shape[-1])
    target_frame_rows = (latent_h // 2) * (latent_w // 2)

    entry_reports: list[dict[str, Any]] = []
    for entry_index, entry in enumerate(positive):
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise TypeError("conditioning entries must contain a tensor and metadata")
        cond, metadata = entry[0], entry[1]
        cond_shape = _shape(cond)
        if cond_shape is None or len(cond_shape) < 2:
            raise ValueError("conditioning tensor must expose a text-sequence dimension")
        if not isinstance(metadata, Mapping):
            raise TypeError("conditioning metadata must be a mapping")
        rows: dict[str, int] = {
            "text": int(cond_shape[1]),
            "keyframe_video": 0,
            "keyframe_audio": 0,
            "reference_visual": 0,
            "reference_audio": 0,
            "target_audio": audio_t * 2,
            "target_video": latent_t * target_frame_rows,
        }
        blocks: dict[str, int] = {
            "keyframe_video": 0,
            "keyframe_audio": 0,
            "reference_visual": 0,
            "reference_audio": 0,
        }
        keyframes = metadata.get("minimax_keyframes") or []
        references = metadata.get("minimax_refs") or []
        if not isinstance(keyframes, (list, tuple)):
            raise TypeError("minimax_keyframes must be a list or tuple")
        if not isinstance(references, (list, tuple)):
            raise TypeError("minimax_refs must be a list or tuple")
        for keyframe in keyframes:
            if not isinstance(keyframe, Mapping):
                raise TypeError("every minimax_keyframes entry must be a mapping")
            video_shape = _shape(_guide_video(keyframe.get("latent")))
            audio_shape = _shape(keyframe.get("audio_latent"))
            if video_shape is not None:
                if len(video_shape) != 5:
                    raise ValueError("H3 keyframe video latent must be 5D")
                rows["keyframe_video"] += int(video_shape[2]) * target_frame_rows
                blocks["keyframe_video"] += 1
            if audio_shape is not None:
                if len(audio_shape) != 4:
                    raise ValueError("H3 keyframe audio latent must be 4D")
                rows["keyframe_audio"] += int(audio_shape[-1]) * 2
                blocks["keyframe_audio"] += 1
        for reference in references:
            if not isinstance(reference, Mapping):
                raise TypeError("every minimax_refs entry must be a mapping")
            kind = reference.get("kind")
            if kind == "image":
                ref_h, ref_w = _reference_geometry(reference)
                rows["reference_visual"] += ((ref_h + 1) // 2) * ((ref_w + 1) // 2)
                blocks["reference_visual"] += 1
            elif kind == "audio":
                ref_audio_t = int(reference.get("ref_audio_t", 0))
                rows["reference_audio"] += ref_audio_t * 2
                blocks["reference_audio"] += int(ref_audio_t > 0)
            elif kind in ("video", "video_audio"):
                ref_h, ref_w = _reference_geometry(reference)
                latent_ref_t = int(reference.get("latent_t", 0))
                ref_audio_t = int(reference.get("ref_audio_t", 0))
                rows["reference_visual"] += (
                    latent_ref_t * ((ref_h + 1) // 2) * ((ref_w + 1) // 2)
                )
                rows["reference_audio"] += ref_audio_t * 2
                blocks["reference_visual"] += 1
                blocks["reference_audio"] += int(ref_audio_t > 0)
            else:
                raise ValueError(f"unsupported H3 reference kind {kind!r}")
        total_rows = sum(rows.values())
        entry_reports.append(
            {
                "index": entry_index,
                "text_shape": cond_shape,
                "rows": rows,
                "blocks": blocks,
                "total_rows": total_rows,
            }
        )

    active = max(entry_reports, key=lambda item: item["total_rows"])
    total_rows = int(active["total_rows"])
    payload: dict[str, Any] = {
        "schema": H3_PACKED_SEQUENCE_REPORT_SCHEMA,
        "target_frames": target_frames,
        "timeline_origin_frame": int(timeline_origin_frame),
        "target_latent_geometry": {
            "video_tokens": latent_t,
            "height": latent_h,
            "width": latent_w,
            "audio_tokens": audio_t,
            "spatial_patch_rows_per_video_token": target_frame_rows,
        },
        "conditioning_entries": len(entry_reports),
        "active_entry_index": int(active["index"]),
        "total_rows": total_rows,
        "int32_attention_offset_risk": total_rows * 7168 >= 2**31,
        "int32_safe_row_limit": (2**31 - 1) // 7168,
        "estimated_bytes_per_row": calibration,
        "estimated_working_set_bytes": total_rows * calibration,
        "estimate_kind": "linear-calibration-not-a-memory-guarantee",
        "entries": entry_reports,
    }
    payload["report_hash"] = content_hash(payload)
    return payload
