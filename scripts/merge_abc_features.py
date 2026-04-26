#!/usr/bin/env python3
"""
Step 4: one merged feature YAML per (query, graph, config) triple — **A×B×C files**.

Output YAML top-level key order: **config** fields first, then **graph**, then **query**,
then shared ``num_features`` / ``feature_vector`` (fixed slot order inside the vector).
File names: ``{conf_stem}_{graph_stem}_{query_stem}.yaml`` (stems from ``Path.stem``).

Each YAML contains a **53-D** ``feature_vector`` (one row from ``FeatureMerger``; leading slots
are ``price`` (catalog), ``time`` (0 until filled, e.g. from GridGraph), ``cost`` (0 in merge;
optional label use), then static / symbolic / graph / config) plus
``graph_scalars`` and ``symbolic_families_instantiation`` (query symbolic templates
filled with the paired graph's statistics). If a config file lists multiple
``configurations``, only the **first** row is written so that file count stays exactly
A×B×C (A = configs, B = graph features, C = query features).

Usage (all of ``--query-dir``, ``--graph-dir``, ``--config-dir``, ``--out-dir`` are **required**):

  python scripts/merge_abc_features.py \\
    --query-dir out/query_features/gridgraph \\
    --graph-dir out/graph_features \\
    --config-dir data/conf/out-of-core \\
    --out-dir out/features/out-of-core \\
    --query-stems gridgraph_wcc gridgraph_pr gridgraph_bfs gridgraph_spmv
"""
from __future__ import annotations

import argparse
import re
import sys
from itertools import product
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from autoconfig.utils.feature_merger import FeatureMerger  # noqa: E402


def _safe_stem(path: Path) -> str:
    s = path.stem
    s = re.sub(r"[^\w.\-]+", "_", s, flags=re.UNICODE)
    return s.strip("_") or "x"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Write one merged-feature YAML per query×graph×config triple"
    )
    p.add_argument(
        "--query-dir",
        type=Path,
        required=True,
        help="Directory of query feature YAMLs (*.yaml)",
    )
    p.add_argument(
        "--graph-dir",
        type=Path,
        required=True,
        help="Directory of graph feature YAMLs",
    )
    p.add_argument(
        "--config-dir",
        type=Path,
        required=True,
        help="Directory of config YAMLs (e.g. conf/out-of-core)",
    )
    p.add_argument("--config-glob", type=str, default="conf_*.yaml")
    p.add_argument(
        "--query-glob",
        type=str,
        default="*.yaml",
        help="Glob under --query-dir for query YAMLs (default *.yaml)",
    )
    p.add_argument(
        "--query-stems",
        nargs="*",
        default=None,
        metavar="STEM",
        help=(
            "If set, keep only query files whose Path.stem is in this list "
            "(e.g. gridgraph_wcc gridgraph_pr). Default: all files from --query-glob."
        ),
    )
    p.add_argument(
        "--graph-recursive",
        action="store_true",
        help="Include **/*.yaml under --graph-dir (e.g. subfolders); default: top-level only",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for {conf}_{graph}_{query}.yaml files",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print each output path",
    )
    args = p.parse_args()

    queries = sorted(args.query_dir.glob(args.query_glob))
    if args.query_stems:
        want = set(args.query_stems)
        queries = [q for q in queries if q.stem in want]
    if args.graph_recursive:
        graphs = sorted({p for p in args.graph_dir.rglob("*.yaml") if p.is_file()})
    else:
        graphs = sorted(args.graph_dir.glob("*.yaml"))
    confs = sorted(args.config_dir.glob(args.config_glob))
    for label, files, d in (
        ("query", queries, args.query_dir),
        ("graph", graphs, args.graph_dir),
        ("config", confs, args.config_dir),
    ):
        if not files:
            print(f"Error: no *.yaml in {label}-dir: {d}", file=sys.stderr)
            return 1

    a, b, c = len(confs), len(graphs), len(queries)
    merger = FeatureMerger()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for conf, g, q in product(confs, graphs, queries):
        r = merger.merge_all(str(q), str(g), str(conf))
        mat = r["feature_matrix"]
        if not mat:
            print(f"Warning: empty matrix for {q} {g} {conf}", file=sys.stderr)
            continue
        row = mat[0]
        if len(mat) > 1 and args.verbose:
            print(
                f"Note: {conf.name} has {len(mat)} configs; using first row only",
                file=sys.stderr,
            )

        qn, gn, cn = _safe_stem(q), _safe_stem(g), _safe_stem(conf)
        out_path = args.out_dir / f"{cn}_{gn}_{qn}.yaml"
        # Top-level key order: conf -> graph -> query -> merged feature block
        doc = {
            "config_file": conf.name,
            "config_stem": cn,
            "config_scalars": r.get("config_scalars", {}),
            "graph_file": g.name,
            "graph_stem": gn,
            "graph_scalars": r.get("graph_scalars", {}),
            "query_file": q.name,
            "query_stem": qn,
            "symbolic_families_instantiation": r.get(
                "symbolic_families_instantiation", {}
            ),
            "num_features": len(r["feature_names"]),
            "feature_names": r["feature_names"],
            "feature_vector": [float(x) for x in row],
            "feature_groups": r.get("feature_groups", {}),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(doc, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        written += 1
        if args.verbose:
            print(out_path)

    expect = a * b * c
    print(f"Done: {written} files (A×B×C = {a}×{b}×{c} = {expect})")
    if written != expect:
        print(f"Warning: count mismatch (expected {expect})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
