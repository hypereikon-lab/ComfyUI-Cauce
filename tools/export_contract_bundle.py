#!/usr/bin/env python3
"""Generate or verify the deterministic CAUCE contract bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cauce.bundle import build_contract_bundle


def encoded_bundle() -> str:
    return json.dumps(build_contract_bundle(ROOT), ensure_ascii=False, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "operations" / "contract-bundle.json",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    expected = encoded_bundle()
    target = args.output.resolve()
    if args.check:
        if not target.is_file() or target.read_text(encoding="utf-8") != expected:
            print(f"stale CAUCE contract bundle: {target}", file=sys.stderr)
            return 1
        print(f"verified: {target}")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(expected, encoding="utf-8")
    temporary.replace(target)
    print(f"wrote: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
