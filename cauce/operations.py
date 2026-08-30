"""Validation and content addressing for semantic CAUCE operation contracts."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

from .contracts import content_hash


OPERATION_SCHEMA = "cauce.operation/1"
CATALOG_SCHEMA = "cauce.operation-catalog/2"
OPERATION_ID = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
OPERATION_FAMILIES = {
    "h3-conditioning-grammar",
    "native-av-state-algebra",
    "decoded-media-algebra",
    "decoded-video-enhancement",
}
OWNERS = {"official-comfy", "vanilla-comfy", "cauce"}
KINDS = {"h3-inference", "decoded-media-transform", "decoded-video-enhancement"}
IMPLEMENTATION_CLASSES = {
    "official-h3",
    "official-comfy",
    "official-h3-with-cauce-primitives",
    "cauce-preprocess-to-official-h3",
    "cauce-and-vanilla-deterministic",
    "official-comfy-with-cauce-planning",
}
ARTIFACT_STATES = {"contract-only", "paired-graphs"}
EVIDENCE_LEVELS = {
    "defined",
    "unit-validated",
    "schema-validated",
    "executes",
    "visually-characterized",
}
VISUAL_VERDICTS = {"unassessed", "accepted", "rejected", "mixed"}


def load_json(path: Path) -> Any:
    """Load one UTF-8 JSON artifact."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def operation_contract_hash(value: Mapping[str, Any]) -> str:
    """Return a deterministic hash for one complete operation contract."""

    return content_hash(value)


def _validate_ports(value: Any, label: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, list) or (label == "outputs" and not value):
        return [f"{label} must be a valid list"]
    names: set[str] = set()
    for port in value:
        if not isinstance(port, dict) or not {"name", "type", "required"} <= set(port):
            errors.append(f"malformed {label} port {port!r}")
            continue
        name = port.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"{label} port requires a name")
        elif name in names:
            errors.append(f"duplicate {label} port {name!r}")
        else:
            names.add(name)
        if not isinstance(port.get("type"), str) or not port["type"]:
            errors.append(f"{label} port {name!r} requires a type")
        if not isinstance(port.get("required"), bool):
            errors.append(f"{label} port {name!r} required must be boolean")
    return errors


def validate_operation_spec(value: Any) -> list[str]:
    """Validate semantic and ownership invariants not covered by JSON parsing."""

    if not isinstance(value, dict):
        return ["operation spec must be an object"]
    errors: list[str] = []
    required = {
        "schema",
        "id",
        "version",
        "title",
        "kind",
        "implementation_class",
        "summary",
        "inputs",
        "outputs",
        "constraints",
        "graph_contract",
        "artifacts",
        "evidence",
    }
    missing = sorted(required - set(value))
    if missing:
        errors.append(f"missing fields {missing}")
    if value.get("schema") != OPERATION_SCHEMA:
        errors.append("invalid operation schema")
    operation_id = value.get("id")
    if not isinstance(operation_id, str) or not OPERATION_ID.fullmatch(operation_id):
        errors.append(f"invalid operation id {operation_id!r}")
    if not isinstance(value.get("version"), int) or value.get("version", 0) < 1:
        errors.append("version must be a positive integer")
    if value.get("kind") not in KINDS:
        errors.append(f"invalid operation kind {value.get('kind')!r}")
    implementation = value.get("implementation_class")
    if implementation not in IMPLEMENTATION_CLASSES:
        errors.append(f"invalid implementation class {implementation!r}")
    errors.extend(_validate_ports(value.get("inputs"), "inputs"))
    errors.extend(_validate_ports(value.get("outputs"), "outputs"))

    stages = value.get("graph_contract")
    stage_owners: set[str] = set()
    if not isinstance(stages, list) or not stages:
        errors.append("graph_contract must be a non-empty list")
    else:
        orders: list[Any] = []
        for stage in stages:
            if not isinstance(stage, dict):
                errors.append(f"graph stage must be an object: {stage!r}")
                continue
            orders.append(stage.get("order"))
            owner = stage.get("owner")
            if owner not in OWNERS:
                errors.append(f"invalid graph owner {owner!r}")
            else:
                stage_owners.add(owner)
            if not isinstance(stage.get("role"), str) or not stage["role"].strip():
                errors.append("every graph stage requires a role")
        if orders != list(range(1, len(stages) + 1)):
            errors.append("graph stage order must be contiguous from one")

    if implementation == "official-h3" and "cauce" in stage_owners:
        errors.append("official-h3 operation cannot claim CAUCE stages")
    if implementation == "official-comfy":
        if "official-comfy" not in stage_owners:
            errors.append("official-comfy operation requires an official-comfy stage")
        if "cauce" in stage_owners:
            errors.append("official-comfy operation cannot claim CAUCE stages")
    if implementation in {
        "official-h3-with-cauce-primitives",
        "cauce-preprocess-to-official-h3",
    } and not {"official-comfy", "cauce"} <= stage_owners:
        errors.append(f"{implementation} requires official-comfy and cauce stages")
    if implementation == "cauce-and-vanilla-deterministic":
        if "cauce" not in stage_owners:
            errors.append("deterministic CAUCE operation requires a cauce stage")
        if "official-comfy" in stage_owners:
            errors.append("deterministic CAUCE operation cannot contain H3 inference")
    if implementation == "official-comfy-with-cauce-planning":
        if not {"official-comfy", "cauce"} <= stage_owners:
            errors.append(
                "official-comfy-with-cauce-planning requires official-comfy and cauce stages"
            )
    if value.get("kind") == "h3-inference" and "official-comfy" not in stage_owners:
        errors.append("h3-inference operation requires an official-comfy stage")
    if value.get("kind") == "decoded-media-transform" and "official-comfy" in stage_owners:
        errors.append("decoded-media-transform cannot contain H3 inference")
    if value.get("kind") == "decoded-video-enhancement" and "official-comfy" not in stage_owners:
        errors.append("decoded-video-enhancement requires an enhancement model stage")

    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("artifacts must be an object")
    else:
        state = artifacts.get("state")
        if state not in ARTIFACT_STATES:
            errors.append(f"invalid artifact state {state!r}")
        pairs = artifacts.get("pairs")
        if not isinstance(pairs, list):
            errors.append("artifact pairs must be a list")
            pairs = []
        if state == "contract-only" and pairs:
            errors.append("contract-only operation cannot claim graph artifacts")
        if state == "paired-graphs" and not pairs:
            errors.append("paired-graphs operation requires at least one graph pair")

        variant_ids = {
            variant.get("id")
            for variant in value.get("variants", [])
            if isinstance(variant, dict) and isinstance(variant.get("id"), str)
        }
        paired_variants: set[str] = set()
        for pair in pairs:
            if not isinstance(pair, dict):
                errors.append(f"malformed graph artifact pair {pair!r}")
                continue
            required_pair_fields = {
                "variant",
                "ui_graph",
                "api_template",
                "ui_graph_hash",
                "api_template_hash",
            }
            if set(pair) != required_pair_fields:
                errors.append("graph artifact pair has unexpected or missing fields")
                continue
            variant = pair.get("variant")
            if variant not in variant_ids:
                errors.append(f"graph artifact pair references unknown variant {variant!r}")
            elif variant in paired_variants:
                errors.append(f"duplicate graph artifact pair for variant {variant!r}")
            else:
                paired_variants.add(variant)
            for field in ("ui_graph", "api_template"):
                path = pair.get(field)
                if not isinstance(path, str) or not path:
                    errors.append(f"graph artifact pair requires {field}")
            for field in ("ui_graph_hash", "api_template_hash"):
                digest = pair.get(field)
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    errors.append(f"graph artifact pair {field} must be lowercase SHA-256")

    evidence = value.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence must be an object")
    else:
        level = evidence.get("level")
        verdict = evidence.get("visual_verdict")
        if level not in EVIDENCE_LEVELS:
            errors.append(f"invalid evidence level {level!r}")
        if verdict not in VISUAL_VERDICTS:
            errors.append(f"invalid visual verdict {verdict!r}")
        if level == "visually-characterized" and verdict == "unassessed":
            errors.append("visually-characterized evidence requires a visual verdict")
        if level != "visually-characterized" and verdict != "unassessed":
            errors.append("visual verdict requires visually-characterized evidence")
        if not isinstance(evidence.get("summary"), str) or not evidence["summary"].strip():
            errors.append("evidence requires a summary")
    return errors


def load_operation_catalog(root: Path) -> dict[str, dict[str, Any]]:
    """Load and validate every operation named by ``operations/catalog.json``."""

    catalog_path = root / "operations" / "catalog.json"
    catalog = load_json(catalog_path)
    if not isinstance(catalog, dict) or catalog.get("schema") != CATALOG_SCHEMA:
        raise ValueError("invalid operation catalog")
    families = catalog.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError("operation catalog must declare families")
    declared_families: set[str] = set()
    for family in families:
        if not isinstance(family, dict) or set(family) != {"id", "title", "summary"}:
            raise ValueError(f"malformed operation family {family!r}")
        family_id = family.get("id")
        if family_id not in OPERATION_FAMILIES:
            raise ValueError(f"unknown operation family {family_id!r}")
        if family_id in declared_families:
            raise ValueError(f"duplicate operation family {family_id!r}")
        if not all(isinstance(family.get(field), str) and family[field].strip() for field in ("title", "summary")):
            raise ValueError(f"operation family {family_id!r} requires title and summary")
        declared_families.add(family_id)
    if declared_families != OPERATION_FAMILIES:
        raise ValueError("operation catalog family set is incomplete")
    entries = catalog.get("operations")
    if not isinstance(entries, list) or not entries:
        raise ValueError("operation catalog must contain entries")
    loaded: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"id", "version", "family", "spec"}:
            raise ValueError(f"malformed catalog entry {entry!r}")
        if entry["family"] not in declared_families:
            raise ValueError(f"operation {entry['id']!r} has unknown family {entry['family']!r}")
        relative = PurePosixPath(str(entry["spec"]))
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "specs":
            raise ValueError(f"unsafe operation spec path {entry['spec']!r}")
        path = root / "operations" / Path(*relative.parts)
        spec = load_json(path)
        errors = validate_operation_spec(spec)
        if errors:
            raise ValueError(f"{path}: {'; '.join(errors)}")
        if spec["id"] != entry["id"] or spec["version"] != entry["version"]:
            raise ValueError(f"catalog entry does not match {path}")
        if spec["id"] in loaded:
            raise ValueError(f"duplicate operation id {spec['id']!r}")
        for pair in spec["artifacts"]["pairs"]:
            for field in ("ui_graph", "api_template"):
                relative_artifact = PurePosixPath(pair[field])
                expected_suffix = ".ui.json" if field == "ui_graph" else ".api.template.json"
                if (
                    relative_artifact.is_absolute()
                    or ".." in relative_artifact.parts
                    or relative_artifact.parts[0] != "artifacts"
                    or not relative_artifact.name.endswith(expected_suffix)
                ):
                    raise ValueError(f"unsafe {field} path {pair[field]!r}")
                artifact_path = root / "operations" / Path(*relative_artifact.parts)
                try:
                    artifact = load_json(artifact_path)
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid graph artifact {artifact_path}: {exc}") from exc
                actual_hash = content_hash(artifact)
                if actual_hash != pair[f"{field}_hash"]:
                    raise ValueError(f"graph artifact hash mismatch for {artifact_path}")
        loaded[spec["id"]] = spec
    spec_paths = {path.resolve() for path in (root / "operations" / "specs").glob("*.json")}
    loaded_paths = {
        (root / "operations" / Path(*PurePosixPath(str(entry["spec"])).parts)).resolve()
        for entry in entries
    }
    if spec_paths != loaded_paths:
        raise ValueError("catalog and operation spec directory differ")
    return loaded
