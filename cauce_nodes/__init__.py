"""Aggregate native CAUCE node mappings without any web UI."""

from . import (
    artifacts,
    audio,
    continuity,
    h3,
    maintenance,
    masks,
    media,
    motion,
    plates,
    profiles,
    samplers,
    seams,
    timeline,
)


MODULES = (
    timeline,
    media,
    motion,
    samplers,
    plates,
    h3,
    masks,
    audio,
    continuity,
    seams,
    artifacts,
    profiles,
    maintenance,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for module in MODULES:
    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
