"""Compatibility aggregation for split H3 AV node bindings."""

from . import av_inspection, av_masks, av_spans, av_spatial

MODULES = (av_inspection, av_spans, av_masks, av_spatial)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for module in MODULES:
    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = tuple(sorted(NODE_CLASS_MAPPINGS))
