#!/usr/bin/env python3
"""Copy merged GPU feature YAMLs where ``time`` is a positive finite value."""
from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import yaml


def _time_ok(vec: list, names: list[str]) -> bool:
    try:
        i = names.index("time")
        t = float(vec[i])
    except (ValueError, TypeError, IndexError):
        return False
    return math.isfinite(t) and t > 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src: Path = args.src
    dst: Path = args.dst
    if not src.is_dir():
        print(f"not a directory: {src}", file=sys.stderr)
        return 1

    copied = 0
    skipped = 0
    for ypath in sorted(src.glob("*.yaml")):
        with open(ypath, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        if not isinstance(doc, dict):
            skipped += 1
            continue
        names = doc.get("feature_names")
        vec = doc.get("feature_vector")
        if not isinstance(names, list) or not isinstance(vec, list):
            skipped += 1
            continue
        row_names = [str(x) for x in names]
        if not _time_ok(vec, row_names):
            skipped += 1
            continue
        if not args.dry_run:
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ypath, dst / ypath.name)
        copied += 1

    print(f"copied={copied} skipped={skipped} dry_run={args.dry_run}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
