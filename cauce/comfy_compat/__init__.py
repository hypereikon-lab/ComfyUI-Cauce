"""Narrow, fail-closed adapters for the ComfyUI runtime surface used by CAUCE.

CAUCE's mathematical modules never import ComfyUI.  All imports of private or
runtime-owned Comfy modules terminate in this package so upstream drift has one
place to detect, describe, and test.
"""

from .capabilities import ComfyCapabilities, probe_comfy_capabilities
from .conditioning import conditioning_set_values
from .filesystem import output_directory
from .h3 import require_arbitrary_h3_guides
from .nested import make_nested_tensor

__all__ = [
    "ComfyCapabilities",
    "conditioning_set_values",
    "make_nested_tensor",
    "output_directory",
    "probe_comfy_capabilities",
    "require_arbitrary_h3_guides",
]
