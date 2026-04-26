#!/usr/bin/env python3
"""
Fill ``time`` and ``cost`` in merged 53-d YAMLs from GridGraph ``STATS.txt``.

Filename pattern::

    conf_XX_<data>_gridgraph_<task>.yaml

reads a table from::

    <gridgraph_root>/<data>_gridgraph_<task>/STATS.txt

If ``STATS.txt`` is missing, times are read from ``conf_*.output`` in that folder
(last ``real <seconds>`` line per file, e.g. from GNU time). In that case
**核/内存** validation is skipped (``STATS`` is absent).

For each file:

* **time** = **``real``** (last column) from ``STATS`` for the matching ``conf`` row.
* **cost** = **price × time**, where **price** is the existing ``feature_vector`` value for
  the **price** column (from config catalog at merge time).

Validates, unless disabled, that **核** and **内存(GB)** in ``STATS`` match
``config_scalars.cpu_cores`` / ``memory_gb`` in the YAML.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import yaml

CONF_ROW_RE = re.compile(r"^\d{1,2}$")
OUTPUT_FNAME_RE = re.compile(
    r"^conf_(\d{1,2})_.+\.output$",
    re.IGNORECASE,
)
REAL_LINE_RE = re.compile(r"^\s*real\s+([0-9.]+(?:e[+-]?\d+)?)\s*$", re.IGNORECASE)
# ``real`` may be last token on a line (GNU time) or after other text, e.g. ``..., 34, real 0.81``
REAL_TOKEN_RE = re.compile(r"\breal\s+([0-9.]+(?:e[+-]?\d+)?)\b", re.IGNORECASE)


def _split_table_line(line: str) -> List[str] | None:
    line = line.strip()
    if not line.startswith("│") or "conf" in line or "┌" in line or "└" in line or "├" in line:
        return None
    parts = [p.strip() for p in line.split("│") if p.strip() != ""]
    if not parts or not CONF_ROW_RE.match(parts[0]):
        return None
    return parts


class StatsRow(NamedTuple):
    conf_id: str
    cpu_cores: int
    memory_gb: int
    real: float


def parse_stats_table(path: Path) -> Dict[str, StatsRow]:
    """Parse all data rows. Expect: conf, 核, 内存(GB), …, real (last)."""
    out: dict[str, StatsRow] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = _split_table_line(line)
        if not parts or len(parts) < 4:
            continue
        k = parts[0].zfill(2)
        cpu = int(float(parts[1]))
        mem = int(float(parts[2]))
        real = float(parts[-1])
        out[k] = StatsRow(conf_id=k, cpu_cores=cpu, memory_gb=mem, real=real)
    return out


def _parse_output_real_line(path: Path) -> Optional[float]:
    """
    Wall time from the last ``real <seconds>`` occurrence (e.g. GNU time stdout/stderr
    on its own line, or a trailing ``..., real 0.81`` in one line before user/sys lines).
    """
    text = path.read_text(encoding="utf-8")
    last: Optional[float] = None
    for line in text.splitlines():
        m = REAL_LINE_RE.match(line.strip())
        if m:
            last = float(m.group(1))
    if last is not None:
        return last
    for m in REAL_TOKEN_RE.finditer(text):
        last = float(m.group(1))
    return last


def parse_output_folder(dir_path: Path) -> Dict[str, StatsRow]:
    """
    If STATS.txt is missing, use ``conf_NN_*_<folder>.output`` files: ``real`` time per conf.
    cpu_cores / memory_gb are set to 0 (not used; validation is skipped for this group).
    """
    out: dict[str, StatsRow] = {}
    for f in sorted(dir_path.glob("conf_*.output")):
        m = OUTPUT_FNAME_RE.match(f.name)
        if not m:
            continue
        k = m.group(1).zfill(2)
        t = _parse_output_real_line(f)
        if t is not None:
            out[k] = StatsRow(
                conf_id=k, cpu_cores=0, memory_gb=0, real=t
            )
    return out


def write_merged_set_time_and_cost(
    path_out: Path, doc: dict, time_val: float, *, cost_is_price_times_time: bool = True
) -> Tuple[float, float]:
    """
    Set ``time`` from STATS; set ``cost`` to ``price * time`` using current ``price`` in vector.
    Returns (time_written, cost_written).
    """
    names = doc.get("feature_names")
    vec = doc.get("feature_vector")
    if not isinstance(names, list) or not isinstance(vec, list):
        raise ValueError("feature_names or feature_vector missing")
    row_names = [str(x) for x in names]
    for col in ("price", "time", "cost"):
        if col not in row_names:
            raise ValueError(
                f"Expected {col!r} in feature_names (53-d layout: price, time, cost, …)"
            )
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
    vec[i_t] = t
    if cost_is_price_times_time:
        vec[i_c] = float(price * t)
    else:
        vec[i_c] = 0.0
    doc["feature_vector"] = vec
    if doc.get("num_features") is not None:
        doc["num_features"] = len(row_names)
    path_out.write_text(
        yaml.dump(doc, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return t, float(vec[i_c])


def _scalars_from_merged_text(raw: str) -> Optional[Tuple[int, int]]:
    try:
        doc = yaml.safe_load(raw)
    except yaml.YAMLError:
        return None
    if not isinstance(doc, dict):
        return None
    cs = doc.get("config_scalars")
    if not isinstance(cs, dict):
        return None
    c = cs.get("cpu_cores")
    m = cs.get("memory_gb")
    if c is None or m is None:
        return None
    return int(float(c)), int(float(m))


# conf_NN_<anything>_gridgraph_<task>.yaml
FNAME_RE = re.compile(
    r"^conf_(\d+)_(.+_gridgraph_[^.]+)\.yaml$",
    re.IGNORECASE,
)


def main() -> None:
    _repo = Path(__file__).resolve().parent.parent
    _default_gg = _repo.parent / "GridGraph"
    p = argparse.ArgumentParser(
        description="Fill `time` from STATS `real` and set `cost = price * time` (price from feature_vector).",
    )
    p.add_argument(
        "--no-cost-formula",
        action="store_true",
        help="Only set time; leave cost as 0",
    )
    p.add_argument(
        "--gridgraph-root",
        type=Path,
        default=_default_gg,
        help="Root with e.g. web-sk_gridgraph_wcc/STATS.txt",
    )
    p.add_argument(
        "--features-dir",
        type=Path,
        default=_repo / "out/features",
        help="Input merged feature YAMLs (conf_*_*_gridgraph_*.yaml)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_repo / "out/train",
        help="Output directory; use same as --features-dir to overwrite in place",
    )
    p.add_argument(
        "--no-validate-cpu-mem",
        action="store_true",
        help="Do not require STATS 核/内存 to match config_scalars in the YAML",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing STATS, empty table, or cpu/mem mismatch (when validation on)",
    )
    a = p.parse_args()
    cost_formula = not a.no_cost_formula
    a.output_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for path in sorted(a.features_dir.glob("conf_*.yaml")):
        m = FNAME_RE.match(path.name)
        if not m:
            continue
        conf_id, folder_key = m.group(1), m.group(2)
        groups[folder_key].append((conf_id.zfill(2), path))

    if not groups:
        print("No conf_*_*_gridgraph_*.yaml files found.", file=sys.stderr)
        sys.exit(1)

    n_written = 0
    n_skipped = 0
    for folder_key in sorted(groups.keys()):
        if "_gridgraph_" not in folder_key:
            continue
        task = folder_key.rsplit("_gridgraph_", 1)[-1]
        gg_sub = a.gridgraph_root / folder_key
        stats_path = gg_sub / "STATS.txt"
        row_map: dict[str, StatsRow]
        from_output_files = False
        if stats_path.is_file():
            try:
                row_map = parse_stats_table(stats_path)
            except (ValueError, OSError) as e:
                print(f"ERROR parsing {stats_path}: {e}", file=sys.stderr)
                if a.strict:
                    sys.exit(1)
                n_skipped += len(groups[folder_key])
                continue
            if not row_map:
                print(f"SKIP: empty table in {stats_path}", file=sys.stderr)
                if a.strict:
                    sys.exit(1)
                n_skipped += len(groups[folder_key])
                continue
        else:
            row_map = parse_output_folder(gg_sub)
            if not row_map:
                msg = f"SKIP: no {stats_path} and no conf_*.output with real in {gg_sub} (task={task!r})"
                if a.strict:
                    print(msg, file=sys.stderr)
                    sys.exit(1)
                print(msg, file=sys.stderr)
                n_skipped += len(groups[folder_key])
                continue
            from_output_files = True
            print(
                f"NOTE: {folder_key}: no STATS.txt; using {len(row_map)} .output file(s) "
                f"for time (skipping 核/内存 check for this group).",
                file=sys.stderr,
            )

        for k, src in sorted(groups[folder_key], key=lambda t: t[1].name):
            if k not in row_map:
                err = f"No STATS row for conf {k} in {stats_path}"
                if a.strict:
                    raise KeyError(err)
                print(f"WARNING: {err} (skip {src.name})", file=sys.stderr)
                n_skipped += 1
                continue

            st = row_map[k]
            raw = src.read_text(encoding="utf-8")
            try:
                doc = yaml.safe_load(raw)
                if not isinstance(doc, dict):
                    raise ValueError("root must be a mapping")
            except (yaml.YAMLError, ValueError) as e:
                print(f"ERROR parsing {src}: {e}", file=sys.stderr)
                n_skipped += 1
                continue
            if not a.no_validate_cpu_mem and not from_output_files:
                got = _scalars_from_merged_text(raw)
                if got is None:
                    if a.strict:
                        sys.exit("Cannot read config_scalars from " + str(src))
                    print(f"WARNING: no config_scalars in {src.name}, skip validate", file=sys.stderr)
                else:
                    c_yaml, m_yaml = got
                    if c_yaml != st.cpu_cores or m_yaml != st.memory_gb:
                        msg = (
                            f"cpu/mem mismatch {src.name}: YAML (cpu={c_yaml}, mem={m_yaml}) "
                            f"vs STATS (cpu={st.cpu_cores}, mem={st.memory_gb})"
                        )
                        if a.strict:
                            print(msg, file=sys.stderr)
                            sys.exit(1)
                        print(f"WARNING: {msg} — skip", file=sys.stderr)
                        n_skipped += 1
                        continue
            try:
                dst = a.output_dir / src.name
                tw, cw = write_merged_set_time_and_cost(
                    dst, doc, st.real, cost_is_price_times_time=cost_formula
                )
            except ValueError as e:
                print(f"ERROR {src.name}: {e}", file=sys.stderr)
                if a.strict:
                    sys.exit(1)
                n_skipped += 1
                continue
            extra = f"  cost={cw}" if cost_formula else ""
            print(
                f"Wrote {dst}  task={task!r}  time={tw}{extra}",
                flush=True,
            )
            n_written += 1

    print(
        f"Done. {n_written} file(s) written, {n_skipped} skipped. Output: {a.output_dir}",
        file=sys.stderr,
        flush=True,
    )
    if a.strict and n_skipped:
        sys.exit(1)


if __name__ == "__main__":
    main()
