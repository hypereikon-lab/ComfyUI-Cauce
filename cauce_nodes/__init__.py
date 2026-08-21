"""Aggregate CAUCE legacy-node mappings without any web UI."""

from . import artifacts, audio, continuity, h3, masks, plates, profiles, timeline


MODULES = (timeline, plates, h3, masks, audio, continuity, artifacts, profiles)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for module in MODULES:
    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
