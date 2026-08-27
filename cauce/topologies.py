"""Validation for non-executable operation topology dossiers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

from .operations import load_json, load_operation_catalog


TOPOLOGY_SCHEMA = "cauce.operation-topology/1"
TOPOLOGY_CATALOG_SCHEMA = "cauce.operation-topology-catalog/2"
ARCHETYPE_CATALOG_SCHEMA = "cauce.graph-archetype-catalog/1"
KEY = re.compile(r"^[a-z][a-z0-9_]*$")
ARCHETYPE_ID = re.compile(r"^[a-z][a-z0-9-]*$")


def topology_key(operation_id: str, variant: str) -> str:
    """Return the stable catalog key for one operation topology variant."""

    return f"{operation_id}@{variant}"


def validate_topology(value: Any, operation: dict[str, Any]) -> list[str]:
    if not isinstance(value, dict):
        return ["topology must be an object"]
    errors: list[str] = []
    required = {
        "schema",
        "operation",
        "operation_version",
        "variant",
        "state",
        "basis",
        "nodes",
        "edges",
        "bindings",
        "outputs",
        "live_gates",
    }
    if set(value) != required:
        errors.append("topology has unexpected or missing fields")
    if value.get("schema") != TOPOLOGY_SCHEMA:
        errors.append("invalid topology schema")
    if value.get("operation") != operation["id"]:
        errors.append("topology operation does not match its catalog entry")
    if value.get("operation_version") != operation["version"]:
        errors.append("topology operation version does not match")
    variants = {
        item.get("id")
        for item in operation.get("variants", [])
        if isinstance(item, dict)
    }
    if value.get("variant") not in variants:
        errors.append(f"unknown operation variant {value.get('variant')!r}")
    if value.get("state") != "offline-draft":
        errors.append("topology state must remain offline-draft until paired materialization")
    basis = value.get("basis")
    if not isinstance(basis, list) or not basis:
        errors.append("topology requires one or more source-basis records")

    nodes = value.get("nodes")
    node_keys: set[str] = set()
    node_types: set[str] = set()
    if not isinstance(nodes, list) or not nodes:
        errors.append("topology requires nodes")
        nodes = []
    for node in nodes:
        if not isinstance(node, dict) or not {"key", "owner", "class_type", "role"} <= set(node):
            errors.append(f"malformed topology node {node!r}")
            continue
        key = node.get("key")
        if not isinstance(key, str) or not KEY.fullmatch(key) or key in node_keys:
            errors.append(f"invalid or duplicate topology node key {key!r}")
        else:
            node_keys.add(key)
        if node.get("owner") not in {"official-comfy", "vanilla-comfy", "cauce"}:
            errors.append(f"invalid topology node owner {node.get('owner')!r}")
        class_type = node.get("class_type")
        if class_type is not None and (not isinstance(class_type, str) or not class_type):
            errors.append(f"invalid topology class type {class_type!r}")
        elif isinstance(class_type, str):
            node_types.add(class_type)

    required_types = {
        stage["node_type"]
        for stage in operation["graph_contract"]
        if isinstance(stage.get("node_type"), str) and not stage.get("optional")
    }
    missing_types = sorted(required_types - node_types)
    if missing_types:
        errors.append(f"topology omits required graph-contract node types {missing_types}")

    for edge in value.get("edges", []) if isinstance(value.get("edges"), list) else []:
        if not isinstance(edge, dict) or set(edge) != {"from", "to"}:
            errors.append(f"malformed topology edge {edge!r}")
            continue
        for end in ("from", "to"):
            endpoint = edge[end]
            if not isinstance(endpoint, dict) or set(endpoint) != {"node", "port"}:
                errors.append(f"malformed topology edge endpoint {endpoint!r}")
            elif endpoint["node"] not in node_keys:
                errors.append(f"edge references unknown node {endpoint['node']!r}")

    binding_names: set[str] = set()
    for binding in value.get("bindings", []) if isinstance(value.get("bindings"), list) else []:
        if not isinstance(binding, dict) or not {"name", "type", "target", "required"} <= set(binding):
            errors.append(f"malformed topology binding {binding!r}")
            continue
        name = binding.get("name")
        if not isinstance(name, str) or not KEY.fullmatch(name) or name in binding_names:
            errors.append(f"invalid or duplicate topology binding {name!r}")
        else:
            binding_names.add(name)
        target = binding.get("target")
        if (
            not isinstance(target, dict)
            or set(target) != {"node", "input"}
            or target.get("node") not in node_keys
        ):
            errors.append(f"binding references unknown node: {target!r}")

    topology_outputs: dict[str, str] = {}
    for output in value.get("outputs", []) if isinstance(value.get("outputs"), list) else []:
        if not isinstance(output, dict) or not {"name", "type", "source"} <= set(output):
            errors.append(f"malformed topology output {output!r}")
            continue
        if output.get("name") in topology_outputs:
            errors.append(f"duplicate topology output {output.get('name')!r}")
        elif isinstance(output.get("name"), str) and isinstance(output.get("type"), str):
            topology_outputs[output["name"]] = output["type"]
        source = output.get("source")
        if (
            not isinstance(source, dict)
            or set(source) != {"node", "port"}
            or source.get("node") not in node_keys
        ):
            errors.append(f"output references unknown node: {source!r}")
    contract_outputs = {item["name"]: item["type"] for item in operation["outputs"]}
    if topology_outputs != contract_outputs:
        errors.append("topology outputs must exactly match the operation contract")

    live_gates = value.get("live_gates")
    if not isinstance(live_gates, list) or not live_gates:
        errors.append("topology requires explicit live gates")
    return errors


def load_topology_catalog(root: Path) -> dict[str, dict[str, Any]]:
    operations = load_operation_catalog(root)
    catalog = load_json(root / "operations" / "topologies" / "catalog.json")
    if not isinstance(catalog, dict) or catalog.get("schema") != TOPOLOGY_CATALOG_SCHEMA:
        raise ValueError("invalid topology catalog")
    entries = catalog.get("topologies")
    if not isinstance(entries, list) or not entries:
        raise ValueError("topology catalog must contain entries")
    loaded: dict[str, dict[str, Any]] = {}
    paths: set[Path] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"operation", "variant", "path"}:
            raise ValueError(f"malformed topology catalog entry {entry!r}")
        operation_id = entry["operation"]
        variant = entry["variant"]
        key = topology_key(operation_id, variant)
        if operation_id not in operations or key in loaded:
            raise ValueError(f"unknown or duplicate topology variant {key!r}")
        relative = PurePosixPath(str(entry["path"]))
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "plans":
            raise ValueError(f"unsafe topology path {entry['path']!r}")
        path = root / "operations" / "topologies" / Path(*relative.parts)
        topology = load_json(path)
        errors = validate_topology(topology, operations[operation_id])
        if topology.get("variant") != variant:
            errors.append("topology variant does not match its catalog entry")
        if errors:
            raise ValueError(f"{path}: {'; '.join(errors)}")
        loaded[key] = topology
        paths.add(path.resolve())
    expected = {
        path.resolve()
        for path in (root / "operations" / "topologies" / "plans").glob("*.json")
    }
    if {value["operation"] for value in loaded.values()} != set(operations):
        raise ValueError("topology catalog must cover every operation at least once")
    if paths != expected:
        raise ValueError("topology catalog and plan directory differ")
    return loaded


def topology_signature(topology: dict[str, Any]) -> str:
    """Hash graph structure while excluding bindings, roles, and live state."""

    structure = {
        "nodes": sorted(
            (
                {
                    "key": node["key"],
                    "class_type": node["class_type"],
                    "owner": node["owner"],
                }
                for node in topology["nodes"]
            ),
            key=lambda node: node["key"],
        ),
        "edges": sorted(
            topology["edges"],
            key=lambda edge: (
                edge["from"]["node"],
                edge["from"]["port"],
                edge["to"]["node"],
                edge["to"]["port"],
            ),
        ),
    }
    encoded = json.dumps(structure, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_archetype_catalog(root: Path) -> dict[str, dict[str, Any]]:
    """Load structural graph archetypes and prove exact topology coverage."""

    topologies = load_topology_catalog(root)
    catalog = load_json(root / "operations" / "archetypes" / "catalog.json")
    if not isinstance(catalog, dict) or catalog.get("schema") != ARCHETYPE_CATALOG_SCHEMA:
        raise ValueError("invalid graph-archetype catalog")
    entries = catalog.get("archetypes")
    if not isinstance(entries, list) or not entries:
        raise ValueError("graph-archetype catalog must contain entries")

    loaded: dict[str, dict[str, Any]] = {}
    covered: set[str] = set()
    signatures: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "id",
            "topology_signature",
            "topology_keys",
        }:
            raise ValueError(f"malformed graph archetype {entry!r}")
        archetype_id = entry["id"]
        keys = entry["topology_keys"]
        signature = entry["topology_signature"]
        if (
            not isinstance(archetype_id, str)
            or not ARCHETYPE_ID.fullmatch(archetype_id)
            or archetype_id in loaded
        ):
            raise ValueError(f"invalid or duplicate graph-archetype id {archetype_id!r}")
        if not isinstance(keys, list) or not keys or keys != sorted(set(keys)):
            raise ValueError(f"graph archetype {archetype_id!r} needs sorted unique topology keys")
        if any(key not in topologies for key in keys):
            raise ValueError(f"graph archetype {archetype_id!r} references an unknown topology")
        if covered.intersection(keys):
            raise ValueError(f"graph archetype {archetype_id!r} repeats a topology")
        actual = {topology_signature(topologies[key]) for key in keys}
        if actual != {signature}:
            raise ValueError(f"graph archetype {archetype_id!r} mixes graph structures")
        if signature in signatures:
            raise ValueError(f"graph archetype {archetype_id!r} duplicates a structural signature")
        covered.update(keys)
        signatures.add(signature)
        loaded[archetype_id] = entry
    if covered != set(topologies):
        raise ValueError("graph-archetype catalog must cover every topology exactly once")
    return loaded
