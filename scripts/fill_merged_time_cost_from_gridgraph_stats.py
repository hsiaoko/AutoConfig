#!/usr/bin/env python3
"""
Fill ``time`` and ``cost`` (price * time) in merged feature YAMLs from GridGraph STATS.txt.

Expects paths like::
  {exp_root}/{dataset}/{dataset}_gridgraph_{sssp|coloring}/STATS.txt

Updates only files whose query_stem is gridgraph_sssp or gridgraph_coloring.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

_TASK_SUFFIX = {
    "gridgraph_sssp": "sssp",
    "gridgraph_coloring": "coloring",
}


def parse_stats_table(path: Path) -> dict[int, float]:
    """Map conf index (1..N) -> real time (seconds) from box-drawing table."""
    out: dict[int, float] = {}
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if "│" not in line:
            continue
        s = line.strip()
        if s.startswith(("┌", "├", "└", "│ conf")):
            continue
        parts = [p.strip() for p in line.split("│")]
        parts = [p for p in parts if p != ""]
        if len(parts) < 6:
            continue
        m = re.match(r"^(\d+)$", parts[0])
        if not m:
            continue
        try:
            conf_i = int(m.group(1))
            real_s = parts[5].replace("—", "-")
            if real_s in ("-", ""):
                continue
            out[conf_i] = float(real_s)
        except (ValueError, IndexError):
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--merged-dir",
        type=Path,
        required=True,
        help="Directory of merged *.yaml (e.g. out/features/out-of-core)",
    )
    ap.add_argument(
        "--gridgraph-exp",
        type=Path,
        required=True,
        help="GridGraph exp root (e.g. .../GridGraph/exp)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cache: dict[tuple[str, str], dict[int, float]] = {}
    updated = 0
    skipped = 0

    for ypath in sorted(args.merged_dir.glob("*.yaml")):
        with open(ypath, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        if not isinstance(doc, dict):
            continue
        qstem = doc.get("query_stem", "")
        if qstem not in _TASK_SUFFIX:
            continue
        gstem = str(doc.get("graph_stem", ""))
        cstem = str(doc.get("config_stem", ""))
        m = re.match(r"conf_(\d+)$", cstem)
        if not gstem or not m:
            print(f"skip (bad stems): {ypath.name}", file=sys.stderr)
            skipped += 1
            continue
        conf_i = int(m.group(1))
        task = _TASK_SUFFIX[qstem]
        key = (gstem, task)
        if key not in cache:
            stats_path = (
                args.gridgraph_exp
                / gstem
                / f"{gstem}_gridgraph_{task}"
                / "STATS.txt"
            )
            if not stats_path.is_file():
                print(f"missing STATS: {stats_path}", file=sys.stderr)
                skipped += 1
                continue
            cache[key] = parse_stats_table(stats_path)
        times = cache[key]
        if conf_i not in times:
            print(f"no row conf {conf_i} in {gstem} {task}", file=sys.stderr)
            skipped += 1
            continue
        t = times[conf_i]
        vec = doc.get("feature_vector")
        if not isinstance(vec, list) or len(vec) < 3:
            print(f"bad feature_vector: {ypath}", file=sys.stderr)
            skipped += 1
            continue
        price = float(vec[0])
        vec[1] = round(float(t), 6)
        vec[2] = round(price * t, 6)
        doc["feature_vector"] = vec
        if not args.dry_run:
            with open(ypath, "w", encoding="utf-8") as f:
                yaml.dump(
                    doc,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
        updated += 1

    print(f"Updated {updated} YAMLs (dry_run={args.dry_run}); skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
