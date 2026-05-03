#!/usr/bin/env python3
"""
Smoke: train on merged rows from 3 per-file conf YAMLs, then ``recommend_top_k`` with
**``--config <dir>``** (one ``.yaml`` per configuration). Asserts each ranked item has ``source_path``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoconfig.conf_recommend.recommend import _load_config_dir, recommend_top_k
from autoconfig.merged import train_bayesian_cost_from_merged_yamls
from autoconfig.utils.feature_merger import FeatureMerger


def _one_conf_file(path: Path, cpu: int) -> None:
    path.write_text(
        f"""catalog:
- price: 1
  cpu_cores: 4
  memory_gb: 8
  storage_gb: 32
  num_gpus: 0
configurations:
- node_id: 1
  resource:
    cpu_cores: {cpu}
    memory_gb: 8
    storage_gb: 32
    num_gpus: 0
    grid_size: 0
    block_size: 0
""",
        encoding="utf-8",
    )


def _q_g() -> tuple[dict, dict]:
    q = {
        "query_features": {
            "static": {f"static_{k}": 0.0 for k in [
                "loop_count", "max_loop_depth", "branch_count", "variable_count",
                "recursion_count", "atomic_op_count", "sync_count", "explicit_parallel_flag",
            ]},
            "symbolic": {
                "format_version": 2,
                "families": {k: {"detected": False} for k in [
                    "vscan", "escan", "fscan", "rexp", "atom", "comm",
                ]},
            },
        }
    }
    g = {
        "graph_features": {
            "basic": {"num_vertices": 10, "num_edges": 20},
            "degree": {"avg": 4.0, "skew": 1.0, "max": 5, "min": 1, "std": 0.0},
            "structure": {"diameter": 7, "clustering_coeff": 0.0, "num_components": 1},
            "partition": {
                "num_partitions": 1, "boundary_vertices": 0, "boundary_degree_sum": 0.0,
                "avg_partition_size": 0.0, "partition_size_std": 0.0,
                "edge_cut_ratio": 0.0, "balance": 1.0,
            },
        }
    }
    return q, g


def main() -> int:
    q, g = _q_g()
    merger = FeatureMerger()
    with tempfile.TemporaryDirectory(prefix="autoconfig_recommend_dir_") as td:
        tdp = Path(td)
        _one_conf_file(tdp / "01_low_cpu.yaml", cpu=2)
        _one_conf_file(tdp / "02_mid_cpu.yaml", cpu=4)
        _one_conf_file(tdp / "03_high_cpu.yaml", cpu=8)
        cmerged, _ = _load_config_dir(tdp)
        cfg_rows = merger.extract_config_features(cmerged)
        mat, names = merger.merge(q, g, cfg_rows)
        names = list(names)
        n = int(mat.shape[0])
        train_d = tdp / "train"
        train_d.mkdir()
        for i in range(n):
            vec = [float(x) for x in mat[i].tolist()]
            vec[0] = 0.0
            vec[1] = float(1.0 + 10.0 * i)
            vec[2] = 0.0
            doc = {
                "num_features": len(names),
                "feature_names": names,
                "feature_vector": vec,
            }
            (train_d / f"sample_{i:02d}.yaml").write_text(
                yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        out_m = tdp / "models"
        train_bayesian_cost_from_merged_yamls(
            train_d,
            out_m,
            test_split=0.0,
            pattern="*.yaml",
            model_basename="smoke_dir_merged",
            target_name="time",
            y_axis=1,
            n_iter=50,
            verbose=True,
        )
        pkl = out_m / "smoke_dir_merged.pkl"
        r = recommend_top_k(pkl, q, g, tdp, k=2)

    print("\n=== recommend_top_k(…, config_dir) — each item has source_path ===\n")
    for it in r.items:
        if not it.source_path:
            print("ERROR: expected source_path for directory input", file=sys.stderr)
            return 1
        print(
            f"  rank {it.rank}  pred={it.predicted:.6f}  idx={it.config_index}\n"
            f"    {it.source_path}\n"
        )
    print("OK — directory smoke test finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
