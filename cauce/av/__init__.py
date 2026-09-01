"""Typed, ComfyUI-independent algebra over packed MiniMax H3 AV state."""

from .backend import TensorBackend, backend_for
from .types import AVLatent, AVSpan, AVSpanDescriptor, AVWindowLayout, NestedFactory

__all__ = [
    "AVLatent",
    "AVSpan",
    "AVSpanDescriptor",
    "AVWindowLayout",
    "NestedFactory",
    "TensorBackend",
    "backend_for",
]
