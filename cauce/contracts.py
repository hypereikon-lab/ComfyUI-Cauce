"""Small serialized contracts shared by CAUCE operations."""

from __future__ import annotations

import hashlib
import json
from typing import Any


SEAM_SCHEMA = "cauce.seam/1"


def canonical_json(value: Any) -> str:
    """Serialize a value deterministically for reports and content hashes."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    """Return the SHA-256 hash of a deterministic JSON representation."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
