#!/usr/bin/env python3
"""
Sample system configuration YAMLs (one file per configuration), merge-style.

* **lhs** — Latin hypercube in the same 8D unit space as
  :file:`data/conf/build_ten_conf.py` (reuses that module's
  :func:`_build_resource` for constraints: ``num_gpus==0 => grid=block=0``, etc.).

* **grid** — Full factorial in ``[0,1)^8`` on a per-axis level count ``k`` (minimal ``k`` with
  ``k**8 >= n``), then take the first ``n`` index tuples in row-major order (deterministic;
  for large ``n`` consider ``--method lhs`` instead).

Run from repo root:

  ./scripts/generate_confs.sh -n 50 -o exp/conf/out-of-core --method lhs
  .venv/bin/python scripts/generate_confs.py -n 20 -o /tmp/cfgs --method grid --seed 0
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

import numpy as np

# Repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoconfig.utils.cloud_pricing_research import (  # noqa: E402
    fetch_cloud_pricing_perplexity,
    zero_cloud_pricing,
)

_BTC_PATH = ROOT / "data" / "conf" / "build_ten_conf.py"


def _load_btc():
    spec = importlib.util.spec_from_file_location("_build_ten_conf", _BTC_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {_BTC_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _unit_cube_rows_grid(n: int, d: int, seed: int) -> np.ndarray:
    """
    Return ``(n, d)`` with values in (0,1) from a product grid, row-major on indices.
    Chooses the smallest ``k`` with ``k**d >= n`` (``k`` at least 2 if ``n > 1``).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if d < 1:
        raise ValueError("d must be >= 1")
    k = 1
    while k**d < n:
        k += 1
    k = max(2, k) if n > 1 else 1
    axes: List[np.ndarray] = []
    for _ in range(d):
        if k == 1:
            axes.append(np.array([0.5], dtype=np.float64))
        else:
            axes.append(np.linspace(1e-6, 1.0 - 1e-6, k, dtype=np.float64))
    # Row-major: last index runs fastest
    grid_idx = list(np.ndindex((k,) * d))
    if len(grid_idx) < n:
        raise RuntimeError("internal: grid too small")
    grid_idx = grid_idx[:n]
    U = np.zeros((n, d), dtype=np.float64)
    for r, idx in enumerate(grid_idx):
        for j in range(d):
            U[r, j] = float(axes[j][idx[j]] if k > 1 else axes[j][0])
    return U


def _write_one(
    path: Path,
    resource: Dict[str, Any],
    sample_index: int,
    meta_extras: Dict[str, Any],
    with_cloud: bool,
    api_key: str,
    model: str,
) -> None:
    s = {k: v for k, v in resource.items() if v is not None}
    body_meta: Dict[str, Any] = {**meta_extras, "sample_index": sample_index}
    if with_cloud and (
        (api_key or "").strip() or __import__("os").environ.get("PERPLEXITY_API_KEY")
    ):
        pr = fetch_cloud_pricing_perplexity(
            s, api_key=(api_key or None), model=(model or None) or None
        )
        body_meta["cloud_pricing"] = pr
    else:
        body_meta["cloud_pricing"] = zero_cloud_pricing(
            "未联网询价；价格填 0（需要时加 --with-cloud-pricing 并设置 PERPLEXITY_API_KEY）"
        )
    body = {
        "catalog": [{"price": 0}, copy.deepcopy(s)],
        "configurations": [{"node_id": 1, "resource": copy.deepcopy(s)}],
        "metadata": body_meta,
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(body, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def main() -> None:
    btc = _load_btc()
    p = argparse.ArgumentParser(
        description="Generate N merge-ready config YAMLs (one per file); LHS or grid in unit 8-cube"
    )
    p.add_argument("-n", "--n-conf", type=int, required=True, help="Number of config files")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output directory (created if missing)",
    )
    p.add_argument(
        "--method",
        choices=("lhs", "grid"),
        default="lhs",
        help="lhs: Latin hypercube; grid: product grid in unit 8D (see --help text)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prefix", type=str, default="conf", help="Filename prefix: {prefix}_01.yaml")
    p.add_argument(
        "--gpu-fraction",
        type=float,
        default=0.5,
        help="(LHS/ grid → _build_resource) u3 branch: fraction of rows that may go GPU",
    )
    p.add_argument("--cpu-min", type=int, default=1)
    p.add_argument("--cpu-max", type=int, default=128)
    p.add_argument("--mem-min", type=int, default=1)
    p.add_argument("--mem-max", type=int, default=128)
    p.add_argument("--storage-min", type=int, default=32)
    p.add_argument("--storage-max", type=int, default=4096)
    p.add_argument("--grid-min", type=int, default=1)
    p.add_argument("--grid-max", type=int, default=128)
    p.add_argument("--block-min", type=int, default=1)
    p.add_argument("--block-max", type=int, default=128)
    p.add_argument("--num-gpus-min", type=int, default=0)
    p.add_argument("--num-gpus-max", type=int, default=1)
    p.add_argument("--gpu-mem-min", type=int, default=8)
    p.add_argument("--gpu-mem-max", type=int, default=64)
    p.add_argument("--sm-base", type=int, default=2560)
    p.add_argument("--with-cloud-pricing", action="store_true")
    p.add_argument("--perplexity-api-key", type=str, default="")
    p.add_argument("--perplexity-model", type=str, default="")
    p.add_argument("--cloud-pricing-delay", type=float, default=1.0)
    args = p.parse_args()

    n = int(args.n_conf)
    if n < 1:
        p.error("-n / --n-conf must be >= 1")
    d = 8
    if args.method == "lhs":
        U = btc._lhs_unit_matrix(n, d, int(args.seed))
    else:
        U = _unit_cube_rows_grid(n, d, int(args.seed))

    out_dir: Path = args.output
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {
        "sampling_method": f"{args.method}_8d" + ("_latin_hypercube" if args.method == "lhs" else "_product_grid"),
        "seed": int(args.seed),
        "num_samples": n,
        "num_dimensions": d,
        "method": args.method,
        "ranges": {
            "cpu_cores": [args.cpu_min, args.cpu_max],
            "memory_gb": [args.mem_min, args.mem_max],
            "storage_gb": [args.storage_min, args.storage_max],
            "grid_size": [args.grid_min, args.grid_max],
            "block_size": [args.block_min, args.block_max],
            "num_gpus": [args.num_gpus_min, args.num_gpus_max],
            "gpu_memory_gb": [args.gpu_mem_min, args.gpu_mem_max],
        },
        "gpu_branch_u3_gte_1_minus_fraction": 1.0 - float(args.gpu_fraction),
        "generator_script": "scripts/generate_confs.py",
        "constraints": [
            "num_gpus == 0 => grid_size == 0 and block_size == 0",
            "num_gpus > 0 => grid_size > 0 and block_size > 0",
        ],
    }
    if not (0.0 <= args.gpu_fraction <= 1.0):
        p.error("--gpu-fraction must be in [0,1]")

    if args.num_gpus_max < args.num_gpus_min:
        p.error("--num-gpus-max must be >= --num-gpus-min")
    for lo, hi in [
        (args.cpu_min, args.cpu_max),
        (args.mem_min, args.mem_max),
    ]:
        if lo > hi:
            p.error("cpu/mem min must be <= max")

    paths: List[Path] = []
    for i in range(n):
        s = btc._build_resource(
            U[i],
            cpu_lo=args.cpu_min,
            cpu_hi=args.cpu_max,
            mem_lo=args.mem_min,
            mem_hi=args.mem_max,
            storage_lo=args.storage_min,
            storage_hi=args.storage_max,
            grid_lo=args.grid_min,
            grid_hi=args.grid_max,
            block_lo=args.block_min,
            block_hi=args.block_max,
            num_gpus_lo=args.num_gpus_min,
            num_gpus_hi=args.num_gpus_max,
            gpu_mem_lo=args.gpu_mem_min,
            gpu_mem_hi=args.gpu_mem_max,
            sm_base=args.sm_base,
            gpu_fraction=args.gpu_fraction,
        )
        name = f"{args.prefix}_{i + 1:02d}.yaml"
        pth = out_dir / name
        _write_one(
            pth,
            s,
            sample_index=i + 1,
            meta_extras=copy.deepcopy(meta),
            with_cloud=bool(args.with_cloud_pricing),
            api_key=str(args.perplexity_api_key or ""),
            model=str(args.perplexity_model or ""),
        )
        if (
            args.with_cloud_pricing
            and i + 1 < n
            and float(args.cloud_pricing_delay) > 0
        ):
            time.sleep(float(args.cloud_pricing_delay))
        paths.append(pth)

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
