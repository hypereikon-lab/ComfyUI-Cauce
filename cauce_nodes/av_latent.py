"""ComfyUI bindings for low-level MiniMax H3 audiovisual-latent operations."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping

from ..cauce.av_latent import (
    allocate_av_window_like,
    append_av_span,
    build_av_span_keyframes,
    extract_av_span,
    inspect_av_latent,
    plan_av_window,
)


CATEGORY = "CAUCE/H3 AV Latent"


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _nested_factory(streams):
    from comfy.nested_tensor import NestedTensor

    return NestedTensor(streams)


def _existing_keyframes(positive):
    if not isinstance(positive, list) or not positive:
        raise TypeError("positive must be a non-empty ComfyUI CONDITIONING list")
    first = positive[0]
    if not isinstance(first, (list, tuple)) or len(first) < 2:
        raise TypeError("positive entries must contain a tensor and metadata mapping")
    metadata = first[1]
    if not isinstance(metadata, Mapping):
        raise TypeError("positive conditioning metadata must be a mapping")
    keyframes = metadata.get("minimax_keyframes", [])
    if not isinstance(keyframes, (list, tuple)):
        raise TypeError("positive minimax_keyframes metadata must be a list or tuple")
    if not all(isinstance(keyframe, Mapping) for keyframe in keyframes):
        raise TypeError("every positive minimax_keyframes entry must be a mapping")
    return list(keyframes)


def _require_native_arbitrary_guides():
    from comfy.ldm.minimax.model import PackedLayout

    if "frame_count" in inspect.signature(PackedLayout.__init__).parameters:
        raise RuntimeError(
            "CAUCE H3 latent guides require a ComfyUI build with arbitrary-frame H3 AV guides"
        )


class CauceH3InspectAVLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "timeline_origin_frame": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10_000_000},
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("frame_count", "video_tokens", "audio_tokens", "report_json")
    FUNCTION = "inspect"
    CATEGORY = CATEGORY
    DESCRIPTION = "Validate and report one packed MiniMax H3 AV latent or aligned window."

    def inspect(self, latent, timeline_origin_frame):
        report = inspect_av_latent(
            latent,
            timeline_origin_frame=int(timeline_origin_frame),
        )
        return (
            int(report["frame_count"]),
            int(report["video_tokens"]),
            int(report["audio_tokens"]),
            _json(report),
        )


class CauceH3PlanAVWindow:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "previous_av_latent": ("LATENT",),
                "overlap_frames": (
                    "INT",
                    {"default": 22, "min": 5, "max": 3600, "step": 17},
                ),
                "extension_frames": (
                    "INT",
                    {"default": 119, "min": 17, "max": 3570, "step": 17},
                ),
            }
        }

    RETURN_TYPES = ("CAUCE_H3_AV_LAYOUT", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "layout",
        "window_frames",
        "window_start_frame",
        "overlap_frames",
        "extension_frames",
        "report_json",
    )
    FUNCTION = "plan"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Resolve exact global frame and video/audio-token boundaries for one fresh H3 AV window."
    )

    def plan(self, previous_av_latent, overlap_frames, extension_frames):
        layout = plan_av_window(
            previous_av_latent,
            overlap_frames=int(overlap_frames),
            extension_frames=int(extension_frames),
        )
        return (
            layout,
            int(layout["window_frame_count"]),
            int(layout["window_start_frame"]),
            int(layout["overlap_frames"]),
            int(layout["extension_frames"]),
            _json(layout),
        )


class CauceH3AllocateAVWindow:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "previous_av_latent": ("LATENT",),
                "layout": ("CAUCE_H3_AV_LAYOUT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "allocate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Allocate a zero H3 AV target using a validated absolute timeline layout."

    def allocate(self, previous_av_latent, layout):
        return (
            allocate_av_window_like(
                previous_av_latent,
                layout,
                nested_factory=_nested_factory,
            ),
        )


class CauceH3ExtractAVSpan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "timeline_origin_frame": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10_000_000},
                ),
                "start_frame": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10_000_000},
                ),
                "frame_count": (
                    "INT",
                    {"default": 22, "min": 1, "max": 10_000_000},
                ),
            }
        }

    RETURN_TYPES = ("CAUCE_H3_AV_SPAN", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("span", "frame_count", "video_tokens", "audio_tokens", "report_json")
    FUNCTION = "extract"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Extract synchronized H3 video/audio tokens at exact global frame boundaries."
    )

    def extract(self, latent, timeline_origin_frame, start_frame, frame_count):
        span = extract_av_span(
            latent,
            timeline_origin_frame=int(timeline_origin_frame),
            start_frame=int(start_frame),
            frame_count=int(frame_count),
        )
        descriptor = span["descriptor"]
        return (
            span,
            int(descriptor["frame_count"]),
            int(descriptor["video_tokens"]),
            int(descriptor["audio_tokens"]),
            _json({
                "schema": span["schema"],
                "descriptor": descriptor,
                "descriptor_hash": span["descriptor_hash"],
            }),
        )


class CauceH3AddAVSpanGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "target_av_latent": ("LATENT",),
                "target_layout": ("CAUCE_H3_AV_LAYOUT",),
                "span": ("CAUCE_H3_AV_SPAN",),
                "target_frame_idx": (
                    "INT",
                    {"default": 0, "min": 0, "max": 9999},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "add"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Add one synchronized native H3 AV latent span to conditioning at an explicit frame."
    )

    def add(self, positive, target_av_latent, target_layout, span, target_frame_idx):
        _require_native_arbitrary_guides()
        keyframes = build_av_span_keyframes(
            _existing_keyframes(positive),
            span,
            target_av_latent,
            target_layout,
            target_frame_idx=int(target_frame_idx),
        )
        import node_helpers

        return (node_helpers.conditioning_set_values(positive, {"minimax_keyframes": keyframes}),)


class CauceH3AppendAVSpan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_av_latent": ("LATENT",),
                "span": ("CAUCE_H3_AV_SPAN",),
            }
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "total_frames")
    FUNCTION = "append"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Append one globally contiguous H3 AV span; performs no sampling or overlap policy."
    )

    def append(self, base_av_latent, span):
        return append_av_span(
            base_av_latent,
            span,
            nested_factory=_nested_factory,
        )


NODE_CLASS_MAPPINGS = {
    "CauceH3InspectAVLatent": CauceH3InspectAVLatent,
    "CauceH3PlanAVWindow": CauceH3PlanAVWindow,
    "CauceH3AllocateAVWindow": CauceH3AllocateAVWindow,
    "CauceH3ExtractAVSpan": CauceH3ExtractAVSpan,
    "CauceH3AddAVSpanGuide": CauceH3AddAVSpanGuide,
    "CauceH3AppendAVSpan": CauceH3AppendAVSpan,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceH3InspectAVLatent": "CAUCE · Inspect H3 AV Latent",
    "CauceH3PlanAVWindow": "CAUCE · Plan H3 AV Window",
    "CauceH3AllocateAVWindow": "CAUCE · Allocate H3 AV Window",
    "CauceH3ExtractAVSpan": "CAUCE · Extract H3 AV Span",
    "CauceH3AddAVSpanGuide": "CAUCE · Add H3 AV Span Guide",
    "CauceH3AppendAVSpan": "CAUCE · Append H3 AV Span",
}
