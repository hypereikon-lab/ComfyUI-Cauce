"""Read-only inspection of MiniMax H3 conditioning metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import content_hash
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
