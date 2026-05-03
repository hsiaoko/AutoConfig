#!/usr/bin/env python3
"""
Build minimal query/graph/multi-config, train a tiny train-merged model, then
:func:`recommend_top_k` — for manual testing of ``conf_recommend`` / ``recommend-conf``.

Run from repo root (after ``pip install -e .``):

  .venv/bin/python scripts/recommend_conf_smoke.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoconfig.conf_recommend import recommend_top_k
from autoconfig.merged import train_bayesian_cost_from_merged_yamls
from autoconfig.utils.feature_merger import FeatureMerger


def _minimal_bundle():
    """Same shape as :func:`tests.test_conf_recommend._minimal_query_graph_config`."""
    query = {
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
    graph = {
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
    n = 3
    configurations = [
        {
            "node_id": i + 1,
            "resource": {
                "cpu_cores": 2 + i * 2,
                "memory_gb": 8,
                "storage_gb": 32,
                "num_gpus": 0,
                "grid_size": 0,
                "block_size": 0,
            },
        }
        for i in range(n)
    ]
    config = {
        "catalog": [{"price": 1.0, "cpu_cores": 4, "memory_gb": 8, "storage_gb": 32, "num_gpus": 0}],
        "configurations": configurations,
    }
    return query, graph, config


def main() -> int:
    merger = FeatureMerger()
    q, g, c = _minimal_bundle()
    cfg_rows = merger.extract_config_features(c)
    mat, names = merger.merge(q, g, cfg_rows)
    names = list(names)
    assert len(names) == 53

    with tempfile.TemporaryDirectory(prefix="autoconfig_recommend_smoke_") as td:
        td = Path(td)
        train_d = td / "train"
        train_d.mkdir()
        n = int(mat.shape[0])
        for i in range(n):
            vec = [float(x) for x in mat[i].tolist()]
            vec[0] = 0.0
            vec[1] = float(1.0 + i * 10.0)
            vec[2] = 0.0
            doc = {
                "num_features": len(names),
                "feature_names": names,
                "feature_vector": vec,
            }
            p = train_d / f"sample_{i:02d}.yaml"
            p.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")

        out_m = td / "models"
        train_bayesian_cost_from_merged_yamls(
            train_d,
            out_m,
            test_split=0.0,
            pattern="*.yaml",
            model_basename="smoke_merged",
            target_name="time",
            y_axis=1,
            n_iter=50,
            verbose=True,
        )
        pkl = out_m / "smoke_merged.pkl"
        r = recommend_top_k(
            pkl,
            q,
            g,
            c,
            k=2,
        )

        print("\n=== recommend_top_k (expect lower predicted time first; labels were 1,11,21) ===")
        print(f"target={r.target_name!r}  k={r.k}")
        for it in r.items:
            print(
                f"  rank {it.rank}  idx={it.config_index}  pred={it.predicted:.6f}  "
                f"cpu_cores={it.configuration.get('resource', {}).get('cpu_cores', '?')}"
            )
    print("\nOK — smoke test finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
