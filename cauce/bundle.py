"""Deterministic compilation of CAUCE's complete portable contract surface."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .contracts import content_hash
from .operations import load_json, load_operation_catalog, load_operation_history
from .topologies import load_archetype_catalog, load_topology_catalog, topology_signature

CONTRACT_BUNDLE_SCHEMA = "cauce.contract-bundle/1"


def load_node_registry(root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Load the extension registry offline without installing or importing ComfyUI."""

    module_name = "_cauce_contract_bundle_plugin"
    path = root / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
        submodule_search_locations=[str(root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CAUCE node registry from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        classes = dict(module.NODE_CLASS_MAPPINGS)
        displays = dict(module.NODE_DISPLAY_NAME_MAPPINGS)
    finally:
        for key in list(sys.modules):
            if key == module_name or key.startswith(module_name + "."):
                sys.modules.pop(key, None)
    if set(classes) != set(displays):
        raise ValueError("CAUCE node class and display-name registries differ")
    return classes, displays


def build_contract_bundle(
    root: Path,
    *,
    node_registry: tuple[dict[str, Any], dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Compile every generic contract from canonical CAUCE sources."""

    root = root.resolve()
    operations = load_operation_catalog(root)
    history = load_operation_history(root)
    topologies = load_topology_catalog(root)
    archetypes = load_archetype_catalog(root)
    classes, displays = node_registry or load_node_registry(root)

    current_operations = [
        {
            "id": operation_id,
            "version": spec["version"],
            "contract_hash": content_hash(spec),
        }
        for operation_id, spec in sorted(operations.items())
    ]
    historical_operations = [
        {
            "id": operation_id,
            "version": version,
            "contract_hash": record["contract_hash"],
            "source_commit": record["source_commit"],
        }
        for (operation_id, version), record in sorted(history.items())
    ]
    topology_records = [
        {
            "key": key,
            "operation": topology["operation"],
            "variant": topology["variant"],
            "signature": topology_signature(topology),
        }
        for key, topology in sorted(topologies.items())
    ]
    node_records = [
        {
            "class_type": class_type,
            "display_name": displays[class_type],
            "category": str(getattr(classes[class_type], "CATEGORY", "")),
        }
        for class_type in sorted(classes)
    ]
    payload: dict[str, Any] = {
        "schema": CONTRACT_BUNDLE_SCHEMA,
        "package_version": __version__,
        "operations": {
            "catalog_hash": content_hash(load_json(root / "operations" / "catalog.json")),
            "current": current_operations,
            "history_catalog_hash": content_hash(
                load_json(root / "operations" / "history" / "catalog.json")
            ),
            "historical": historical_operations,
        },
        "topologies": {
            "catalog_hash": content_hash(
                load_json(root / "operations" / "topologies" / "catalog.json")
            ),
            "entries": topology_records,
        },
        "archetypes": {
            "catalog_hash": content_hash(
                load_json(root / "operations" / "archetypes" / "catalog.json")
            ),
            "entries": [archetypes[key] for key in sorted(archetypes)],
        },
        "nodes": node_records,
    }
    payload["bundle_hash"] = content_hash(payload)
    return payload


def validate_contract_bundle(root: Path, bundle: Any) -> list[str]:
    """Compare a serialized bundle with a fresh canonical compilation."""

    if not isinstance(bundle, dict):
        return ["contract bundle must be an object"]
    try:
        expected = build_contract_bundle(root)
    except Exception as exc:  # noqa: BLE001 - validation returns diagnostics, never raises
        return [f"cannot compile canonical contract bundle: {exc}"]
    return [] if bundle == expected else ["contract bundle differs from canonical CAUCE sources"]
