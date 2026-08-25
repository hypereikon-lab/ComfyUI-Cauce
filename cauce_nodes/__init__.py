"""Aggregate CAUCE's deterministic node mappings."""

from . import assembly, motion, persistence, two_sided_window


MODULES = (assembly, motion, persistence, two_sided_window)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for module in MODULES:
    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
