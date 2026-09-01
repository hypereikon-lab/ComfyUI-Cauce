"""ComfyUI bindings for H3 AV spans."""

from __future__ import annotations

from ..cauce.av_latent import (
    append_av_span,
    build_av_span_keyframes,
    place_av_span,
    replace_av_span,
    split_av_latent,
)
from ._shared import (
    conditioning_set_values,
    existing_h3_keyframes,
    json_report,
    make_nested_tensor,
    require_arbitrary_h3_guides,
)

CATEGORY = "CAUCE/H3 AV Latent"

_json = json_report
_nested_factory = make_nested_tensor
_existing_keyframes = existing_h3_keyframes
_require_native_arbitrary_guides = require_arbitrary_h3_guides


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
        return (conditioning_set_values(positive, {"minimax_keyframes": keyframes}),)


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


NODE_CLASS_MAPPINGS = {
    "CauceH3AddAVSpanGuide": CauceH3AddAVSpanGuide,
    "CauceH3AppendAVSpan": CauceH3AppendAVSpan,
    "CauceH3SplitAVLatent": CauceH3SplitAVLatent,
    "CauceH3PlaceAVSpan": CauceH3PlaceAVSpan,
    "CauceH3ReplaceAVSpan": CauceH3ReplaceAVSpan,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceH3AddAVSpanGuide": "CAUCE · Add H3 AV Span Guide",
    "CauceH3AppendAVSpan": "CAUCE · Append H3 AV Span",
    "CauceH3SplitAVLatent": "CAUCE · Split H3 AV Latent",
    "CauceH3PlaceAVSpan": "CAUCE · Place H3 AV Span",
    "CauceH3ReplaceAVSpan": "CAUCE · Replace H3 AV Span",
}
