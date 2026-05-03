#!/usr/bin/env python3
"""
Fill ``time`` (wall ``real`` from MatrixGraph ``STATS.txt``) and ``cost`` (price × time)
in merged feature YAMLs (e.g. ``exp/features/gpu/*.yaml``).

**Source of truth:** timings are aggregated under MatrixGraph ``exp/processed_gpu`` (one
``STATS.txt`` per dataset × algorithm). Those tables correspond to runs under
``exp/gpu/{dataset}/conf_xx/``; rows list ``grid``, ``block``, ``CPU 核``, ``内存 GB`` which
must match the merged YAML ``config_scalars``.

Layout::

  {processed_root}/{graph_stem}/{graph_stem}_matrixgraph_{task}/STATS.txt

``graph_stem`` is e.g. ``friendster``, ``livejournal``, ``web-sk``, ``patents``.

Matching rules (all must hold when STATS supplies the field):

* ``grid_size``, ``block_size`` vs STATS ``grid``, ``block``
* ``cpu_cores`` vs STATS ``CPU 核`` (8-column tables)
* ``memory_gb`` vs STATS ``内存 GB`` when that cell is not ``—`` / ``-``

Query stem → ``task`` suffix: ``kernel_bfs`` → ``bfs``, ``kernel_wcc`` → ``wcc``,
``kernel_pr`` → ``pagerank``, ``kernel_subiso`` → ``subiso``,
``matrixgraph_diameter`` → ``diameter``, ``matrixgraph_skew`` → ``skew``.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

_QUERY_TO_TASK = {
    "kernel_bfs": "bfs",
    "kernel_wcc": "wcc",
    "kernel_pr": "pagerank",
    "kernel_subiso": "subiso",
    "matrixgraph_diameter": "diameter",
    "matrixgraph_skew": "skew",
}


def _optional_float(cell: str) -> Optional[float]:
    s = cell.replace("—", "-").strip()
    if s in ("-", "", "—"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


@dataclass(frozen=True)
class StatsRow:
    conf_id: str  # zero-padded "01".."50"
    grid: int
    block: int
    cpu_cores: Optional[int]
    memory_gb: Optional[float]
    real: float


def _parse_stats_table(path: Path) -> dict[str, StatsRow]:
    """
    Key by conf_id (zfill 2). Last column is wall ``real`` time.

    **8+ columns** (current MatrixGraph tables): conf, grid, block, CPU 核, 内存 GB,
    <kernel metric>, overhead, real.

    **7 columns** (legacy): conf, grid, block, 内存 GB, …, real — CPU column omitted,
    ``cpu_cores`` left None (only grid/block/mem/real matched).
    """
    out: dict[str, StatsRow] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "│" not in line or "conf" in line.lower() or "┌" in line or "└" in line or "├" in line:
            continue
        parts = [p.strip() for p in line.split("│") if p.strip() != ""]
        if len(parts) < 6:
            continue
        raw_conf = parts[0]
        if not re.match(r"^\d{1,3}$", raw_conf):
            continue
        try:
            grid = int(float(parts[1]))
            block = int(float(parts[2]))
        except (ValueError, IndexError):
            continue

        cpu_cores: Optional[int] = None
        memory_gb: Optional[float] = None
        if len(parts) >= 8:
            c = _optional_float(parts[3])
            if c is not None:
                cpu_cores = int(c)
            memory_gb = _optional_float(parts[4])
        elif len(parts) >= 7:
            memory_gb = _optional_float(parts[3])

        real_v = _optional_float(parts[-1])
        if real_v is None or real_v <= 0.0:
            continue

        k = raw_conf.zfill(2)
        out[k] = StatsRow(
            conf_id=k,
            grid=grid,
            block=block,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            real=float(real_v),
        )
    return out


def _set_time_cost(doc: dict, time_val: float) -> None:
    names = doc.get("feature_names")
    vec = doc.get("feature_vector")
    if not isinstance(names, list) or not isinstance(vec, list):
        raise ValueError("feature_names or feature_vector missing")
    row_names = [str(x) for x in names]
    for col in ("price", "time", "cost"):
        if col not in row_names:
            raise ValueError(f"expected {col!r} in feature_names")
    vec = [float(x) for x in vec]
    if len(vec) < len(row_names):
        vec.extend([0.0] * (len(row_names) - len(vec)))
    if len(vec) != len(row_names):
        raise ValueError("feature_vector length does not match feature_names")
    i_p = row_names.index("price")
    i_t = row_names.index("time")
    i_c = row_names.index("cost")
    price = vec[i_p]
    t = float(time_val)
    vec[i_t] = round(t, 6)
    vec[i_c] = round(float(price * t), 6)
    doc["feature_vector"] = vec
    if doc.get("num_features") is not None:
        doc["num_features"] = len(row_names)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--merged-dir",
        type=Path,
        required=True,
        help="Merged YAML directory (e.g. exp/features/gpu)",
    )
    ap.add_argument(
        "--matrixgraph-processed",
        type=Path,
        default=Path("/home/zhuxk/project/graph/MatrixGraph/exp/processed_gpu"),
        help="MatrixGraph exp/processed_gpu root (STATS.txt aggregate)",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--strict", action="store_true", help="Exit 1 if any file skipped")
    args = ap.parse_args()

    cache: dict[tuple[str, str], dict[str, StatsRow]] = {}
    updated = 0
    skipped = 0

    for ypath in sorted(args.merged_dir.glob("*.yaml")):
        with open(ypath, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        if not isinstance(doc, dict):
            skipped += 1
            continue
        qstem = str(doc.get("query_stem", ""))
        gstem = str(doc.get("graph_stem", ""))
        cstem = str(doc.get("config_stem", ""))
        task = _QUERY_TO_TASK.get(qstem)
        m = re.match(r"^conf_(\d+)$", cstem, re.I)
        if not task or not gstem or not m:
            print(f"skip (bad stems): {ypath.name}", file=sys.stderr)
            skipped += 1
            continue
        conf_id = m.group(1).zfill(2)
        cs = doc.get("config_scalars")
        if not isinstance(cs, dict):
            print(f"skip (no config_scalars): {ypath.name}", file=sys.stderr)
            skipped += 1
            continue
        try:
            yaml_grid = int(float(cs["grid_size"]))
            yaml_block = int(float(cs["block_size"]))
            yaml_mem = float(cs["memory_gb"])
            yaml_cpu = int(float(cs["cpu_cores"]))
        except (KeyError, TypeError, ValueError) as e:
            print(f"skip (config_scalars): {ypath.name} {e}", file=sys.stderr)
            skipped += 1
            continue

        key = (gstem, task)
        if key not in cache:
            stats_path = (
                args.matrixgraph_processed
                / gstem
                / f"{gstem}_matrixgraph_{task}"
                / "STATS.txt"
            )
            if not stats_path.is_file():
                print(f"missing STATS: {stats_path}", file=sys.stderr)
                cache[key] = {}
            else:
                cache[key] = _parse_stats_table(stats_path)

        rows = cache[key]
        st = rows.get(conf_id)
        if st is None:
            print(f"no STATS row conf {conf_id}: {ypath.name}", file=sys.stderr)
            skipped += 1
            continue
        if st.grid != yaml_grid or st.block != yaml_block:
            print(
                f"grid/block mismatch {ypath.name}: YAML ({yaml_grid},{yaml_block}) "
                f"vs STATS ({st.grid},{st.block}) for conf {conf_id}",
                file=sys.stderr,
            )
            skipped += 1
            continue
        if st.cpu_cores is not None and st.cpu_cores != yaml_cpu:
            print(
                f"cpu_cores mismatch {ypath.name}: YAML cpu={yaml_cpu} "
                f"vs STATS cpu={st.cpu_cores}",
                file=sys.stderr,
            )
            skipped += 1
            continue
        if st.memory_gb is not None:
            if int(st.memory_gb) != int(yaml_mem):
                print(
                    f"memory mismatch {ypath.name}: YAML mem={yaml_mem} "
                    f"vs STATS mem={st.memory_gb}",
                    file=sys.stderr,
                )
                skipped += 1
                continue

        try:
            _set_time_cost(doc, st.real)
        except ValueError as e:
            print(f"{ypath.name}: {e}", file=sys.stderr)
            skipped += 1
            continue

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

    print(
        f"Updated {updated} YAMLs (dry_run={args.dry_run}); skipped {skipped}",
        file=sys.stderr,
    )
    if args.strict and skipped:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
