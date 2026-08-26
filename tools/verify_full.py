#!/usr/bin/env python3
"""Run the complete CAUCE suite and fail if tensor dependencies are absent."""

from __future__ import annotations

import compileall
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    try:
        import numpy  # noqa: F401
    except ImportError:
        print(
            "full CAUCE verification requires a Python runtime that already provides NumPy; "
            "do not install or upgrade the laboratory GPU stack for this check",
            file=sys.stderr,
        )
        return 2

    if not compileall.compile_dir(ROOT / "cauce", quiet=1):
        return 1
    if not compileall.compile_dir(ROOT / "cauce_nodes", quiet=1):
        return 1
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.skipped:
        print(
            f"full CAUCE verification refuses {len(result.skipped)} skipped tests",
            file=sys.stderr,
        )
        return 2
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
