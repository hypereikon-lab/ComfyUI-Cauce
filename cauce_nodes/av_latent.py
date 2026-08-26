"""ComfyUI bindings for low-level MiniMax H3 audiovisual-latent operations."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping

from ..cauce.av_latent import (
    apply_av_denoise_interval,
    apply_video_denoise_mask,
    allocate_av_window_like,
    append_av_span,
    build_av_span_keyframes,
    clear_av_denoise_mask,
    extract_av_span,
    expand_av_canvas,
    inspect_av_latent,
    place_av_span,
    plan_av_window,
    replace_av_span,
    split_av_latent,
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


class CauceH3SplitAVLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "cut_frame": (
                    "INT",
                    {"default": 124, "min": 5, "max": 10_000_000, "step": 17},
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "CAUCE_H3_AV_SPAN", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "prefix_latent",
        "suffix_span",
        "prefix_frames",
        "suffix_frames",
        "report_json",
    )
    FUNCTION = "split"
    CATEGORY = CATEGORY
    DESCRIPTION = "Split a complete H3 state into a valid prefix and contiguous suffix span."

    def split(self, latent, cut_frame):
        prefix, suffix, prefix_frames, suffix_frames = split_av_latent(
            latent,
            cut_frame=int(cut_frame),
            nested_factory=_nested_factory,
        )
        report = {
            "schema": "cauce.h3-av-split-report/1",
            "timeline_origin_frame": 0,
            "cut_frame": int(cut_frame),
            "prefix_frames": prefix_frames,
            "suffix_frames": suffix_frames,
            "suffix_descriptor": suffix["descriptor"],
            "suffix_descriptor_hash": suffix["descriptor_hash"],
        }
        return prefix, suffix, prefix_frames, suffix_frames, _json(report)


class CauceH3PlaceAVSpan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_av_latent": ("LATENT",),
                "span": ("CAUCE_H3_AV_SPAN",),
                "timeline_origin_frame": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10_000_000},
                ),
                "target_frame_idx": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10_000_000},
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("latent", "rebased", "report_json")
    FUNCTION = "place"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Copy one synchronized native AV span into a target at an exact frame; "
        "does not choose denoise policy."
    )

    def place(self, target_av_latent, span, timeline_origin_frame, target_frame_idx):
        latent, report = place_av_span(
            target_av_latent,
            span,
            timeline_origin_frame=int(timeline_origin_frame),
            target_frame_idx=int(target_frame_idx),
            nested_factory=_nested_factory,
        )
        return latent, bool(report["rebased"]), _json(report)


class CauceH3SetAVDenoiseInterval:
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
                    {"default": 22, "min": 0, "max": 10_000_000},
                ),
                "frame_count": (
                    "INT",
                    {"default": 119, "min": 1, "max": 10_000_000},
                ),
                "inside_strength_video": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "outside_strength_video": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "inside_strength_audio": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "outside_strength_audio": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "fade_in_frames": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10_000_000},
                ),
                "fade_out_frames": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10_000_000},
                ),
                "curve": (["linear", "smoothstep", "smootherstep"],),
                "combine": (["replace", "maximum", "minimum", "multiply"],),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "report_json")
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Attach continuous per-token H3 video/audio noise masks: 1 generates and 0 preserves."
    )

    def apply(
        self,
        latent,
        timeline_origin_frame,
        start_frame,
        frame_count,
        inside_strength_video,
        outside_strength_video,
        inside_strength_audio,
        outside_strength_audio,
        fade_in_frames,
        fade_out_frames,
        curve,
        combine,
    ):
        masked, report = apply_av_denoise_interval(
            latent,
            timeline_origin_frame=int(timeline_origin_frame),
            start_frame=int(start_frame),
            frame_count=int(frame_count),
            inside_strength_video=float(inside_strength_video),
            outside_strength_video=float(outside_strength_video),
            inside_strength_audio=float(inside_strength_audio),
            outside_strength_audio=float(outside_strength_audio),
            fade_in_frames=int(fade_in_frames),
            fade_out_frames=int(fade_out_frames),
            curve=str(curve),
            combine=str(combine),
            nested_factory=_nested_factory,
        )
        return masked, _json(report)


class CauceH3ApplyVideoDenoiseMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "mask": ("MASK",),
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
                    {"default": 124, "min": 1, "max": 10_000_000},
                ),
                "inside_strength_video": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "outside_strength_video": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "audio_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "combine": (["replace", "maximum", "minimum", "multiply"],),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "report_json")
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Project a static or per-frame continuous MASK onto H3 video tokens; "
        "1 generates and 0 preserves."
    )

    def apply(
        self,
        latent,
        mask,
        timeline_origin_frame,
        start_frame,
        frame_count,
        inside_strength_video,
        outside_strength_video,
        audio_strength,
        combine,
    ):
        masked, report = apply_video_denoise_mask(
            latent,
            mask,
            timeline_origin_frame=int(timeline_origin_frame),
            start_frame=int(start_frame),
            frame_count=int(frame_count),
            inside_strength_video=float(inside_strength_video),
            outside_strength_video=float(outside_strength_video),
            audio_strength=float(audio_strength),
            combine=str(combine),
            nested_factory=_nested_factory,
        )
        return masked, _json(report)


class CauceH3ExpandAVCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "target_width": (
                    "INT",
                    {"default": 1024, "min": 32, "max": 16384, "step": 32},
                ),
                "target_height": (
                    "INT",
                    {"default": 1024, "min": 32, "max": 16384, "step": 32},
                ),
                "offset_x": (
                    "INT",
                    {"default": 0, "min": 0, "max": 16384, "step": 32},
                ),
                "offset_y": (
                    "INT",
                    {"default": 0, "min": 0, "max": 16384, "step": 32},
                ),
                "source_strength_video": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "new_region_strength_video": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "audio_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "timeline_origin_frame": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10_000_000},
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "report_json")
    FUNCTION = "expand"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Place one packed H3 AV state on a larger 32-pixel-aligned canvas and "
        "mark only the new region for generation."
    )

    def expand(
        self,
        latent,
        target_width,
        target_height,
        offset_x,
        offset_y,
        source_strength_video,
        new_region_strength_video,
        audio_strength,
        timeline_origin_frame,
    ):
        expanded, report = expand_av_canvas(
            latent,
            target_width=int(target_width),
            target_height=int(target_height),
            offset_x=int(offset_x),
            offset_y=int(offset_y),
            source_strength_video=float(source_strength_video),
            new_region_strength_video=float(new_region_strength_video),
            audio_strength=float(audio_strength),
            timeline_origin_frame=int(timeline_origin_frame),
            nested_factory=_nested_factory,
        )
        return expanded, _json(report)


class CauceH3ReplaceAVSpan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_av_latent": ("LATENT",),
                "replacement_span": ("CAUCE_H3_AV_SPAN",),
                "timeline_origin_frame": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10_000_000},
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "report_json")
    FUNCTION = "replace"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Replace one globally aligned native AV interval and discard any spent noise mask."
    )

    def replace(self, base_av_latent, replacement_span, timeline_origin_frame):
        latent, report = replace_av_span(
            base_av_latent,
            replacement_span,
            timeline_origin_frame=int(timeline_origin_frame),
            nested_factory=_nested_factory,
        )
        return latent, _json(report)


class CauceH3ClearAVDenoiseMask:
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

    RETURN_TYPES = ("LATENT", "BOOLEAN")
    RETURN_NAMES = ("latent", "removed")
    FUNCTION = "clear"
    CATEGORY = CATEGORY
    DESCRIPTION = "Remove a consumed H3 AV noise mask without changing either latent stream."

    def clear(self, latent, timeline_origin_frame):
        return clear_av_denoise_mask(
            latent,
            timeline_origin_frame=int(timeline_origin_frame),
        )


NODE_CLASS_MAPPINGS = {
    "CauceH3InspectAVLatent": CauceH3InspectAVLatent,
    "CauceH3PlanAVWindow": CauceH3PlanAVWindow,
    "CauceH3AllocateAVWindow": CauceH3AllocateAVWindow,
    "CauceH3ExtractAVSpan": CauceH3ExtractAVSpan,
    "CauceH3AddAVSpanGuide": CauceH3AddAVSpanGuide,
    "CauceH3AppendAVSpan": CauceH3AppendAVSpan,
    "CauceH3SplitAVLatent": CauceH3SplitAVLatent,
    "CauceH3PlaceAVSpan": CauceH3PlaceAVSpan,
    "CauceH3SetAVDenoiseInterval": CauceH3SetAVDenoiseInterval,
    "CauceH3ApplyVideoDenoiseMask": CauceH3ApplyVideoDenoiseMask,
    "CauceH3ExpandAVCanvas": CauceH3ExpandAVCanvas,
    "CauceH3ReplaceAVSpan": CauceH3ReplaceAVSpan,
    "CauceH3ClearAVDenoiseMask": CauceH3ClearAVDenoiseMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceH3InspectAVLatent": "CAUCE · Inspect H3 AV Latent",
    "CauceH3PlanAVWindow": "CAUCE · Plan H3 AV Window",
    "CauceH3AllocateAVWindow": "CAUCE · Allocate H3 AV Window",
    "CauceH3ExtractAVSpan": "CAUCE · Extract H3 AV Span",
    "CauceH3AddAVSpanGuide": "CAUCE · Add H3 AV Span Guide",
    "CauceH3AppendAVSpan": "CAUCE · Append H3 AV Span",
    "CauceH3SplitAVLatent": "CAUCE · Split H3 AV Latent",
    "CauceH3PlaceAVSpan": "CAUCE · Place H3 AV Span",
    "CauceH3SetAVDenoiseInterval": "CAUCE · Set H3 AV Denoise Interval",
    "CauceH3ApplyVideoDenoiseMask": "CAUCE · Apply H3 Video Denoise Mask",
    "CauceH3ExpandAVCanvas": "CAUCE · Expand H3 AV Canvas",
    "CauceH3ReplaceAVSpan": "CAUCE · Replace H3 AV Span",
    "CauceH3ClearAVDenoiseMask": "CAUCE · Clear H3 AV Denoise Mask",
}
