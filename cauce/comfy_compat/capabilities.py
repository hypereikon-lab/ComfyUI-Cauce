"""Observable ComfyUI/H3 capability probing without importing CAUCE nodes."""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ComfyCapabilities:
    """The small runtime contract CAUCE actually depends on.

    ``None`` means the capability could not be observed.  It never means false.
    This distinction keeps offline design checks from inventing live support.
    """

    schema: str = "cauce.comfy-capabilities/1"
    comfy_version: str | None = None
    nested_tensor: bool = False
    packed_layout: bool = False
    arbitrary_h3_guides: bool | None = None
    conditioning_set_values: bool = False
    output_directory: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _distribution_version() -> str | None:
    for name in ("comfyui", "ComfyUI"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def probe_comfy_capabilities() -> ComfyCapabilities:
    """Inspect the exact runtime hooks used by CAUCE and return, never raise."""

    nested = packed = conditioning = paths = False
    arbitrary: bool | None = None
    failures: list[str] = []
    try:
        module = importlib.import_module("comfy.nested_tensor")
        nested = callable(getattr(module, "NestedTensor", None))
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - diagnostic probe
        failures.append(f"nested_tensor:{type(exc).__name__}")
    try:
        module = importlib.import_module("comfy.ldm.minimax.model")
        packed_layout = getattr(module, "PackedLayout", None)
        packed = inspect.isclass(packed_layout)
        if packed:
            parameters = inspect.signature(packed_layout.__init__).parameters
            arbitrary = "frame_count" not in parameters and {
                "latent_t",
                "audio_t",
                "keyframes",
            } <= set(parameters)
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - diagnostic probe
        failures.append(f"packed_layout:{type(exc).__name__}")
    try:
        module = importlib.import_module("node_helpers")
        conditioning = callable(getattr(module, "conditioning_set_values", None))
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - diagnostic probe
        failures.append(f"conditioning:{type(exc).__name__}")
    try:
        module = importlib.import_module("folder_paths")
        paths = callable(getattr(module, "get_output_directory", None))
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - diagnostic probe
        failures.append(f"folder_paths:{type(exc).__name__}")
    return ComfyCapabilities(
        comfy_version=_distribution_version(),
        nested_tensor=nested,
        packed_layout=packed,
        arbitrary_h3_guides=arbitrary,
        conditioning_set_values=conditioning,
        output_directory=paths,
        error=",".join(failures) or None,
    )
