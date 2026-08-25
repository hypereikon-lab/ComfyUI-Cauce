"""Aggregate CAUCE's deterministic node mappings."""

from . import assembly, av_latent, motion, persistence


MODULES = (assembly, av_latent, motion, persistence)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for module in MODULES:
    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
