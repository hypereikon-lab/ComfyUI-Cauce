"""Aggregate the stable and research CAUCE node mappings."""

from . import continuity, motion, persistence, research, seams


MODULES = (continuity, seams, motion, persistence, research)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for module in MODULES:
    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
