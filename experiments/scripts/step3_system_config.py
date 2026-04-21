#!/usr/bin/env python3
r"""
Pipeline step 3: generate system / resource configuration candidates (YAML).

Uses Latin Hypercube Sampling over the resource catalog (paper: Sigma_conf).

Output: config_features.yaml (list under ``configurations`` for FeatureMerger).

Example:
  python experiments/scripts/step3_system_config.py -n 20 -o out/config_features.yaml \
    --use-default-catalog
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from autoconfig.utils.config_generator import (  # noqa: E402
    ConfigGenerator,
    generate_default_catalog,
    load_resource_catalog,
    generate_simple_catalog,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=20,
        help="Number of candidate configurations",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=1,
        help="Minimum k (parallel instances)",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=16,
        help="Maximum k",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="out/config_features.yaml",
        help="Output YAML path",
    )
    parser.add_argument(
        "--resource-catalog",
        "-c",
        default=None,
        help="Optional resource catalog YAML",
    )
    parser.add_argument(
        "--use-default-catalog",
        action="store_true",
        help="Use built-in cloud-like catalog",
    )
    parser.add_argument("--cpu", type=int, default=None, help="CPU cores (simple catalog)")
    parser.add_argument("--memory", type=int, default=None, help="Memory GB (simple catalog)")
    parser.add_argument("--gpu", type=int, default=0, help="GPUs (simple catalog)")
    parser.add_argument("--gpu-memory", type=int, default=0, help="GPU memory GB")
    parser.add_argument("--storage", type=int, default=None, help="Storage GB")

    args = parser.parse_args()

    if args.resource_catalog:
        catalog = load_resource_catalog(args.resource_catalog)
    elif args.cpu is not None and args.memory is not None:
        catalog = generate_simple_catalog(
            args.cpu,
            args.memory,
            args.gpu,
            args.gpu_memory,
            args.storage,
        )
    elif args.use_default_catalog:
        catalog = generate_default_catalog()
    else:
        print(
            "Specify one of: --resource-catalog, --use-default-catalog, "
            "or --cpu and --memory",
            file=sys.stderr,
        )
        sys.exit(1)

    generator = ConfigGenerator(catalog)
    result = generator.generate(args.num_samples, (args.k_min, args.k_max))
    meta = result.setdefault("metadata", {})
    meta["pipeline_step"] = 3
    meta["output_file"] = str(args.output)

    generator.save_to_yaml(result, args.output)

    print("Step 3 complete: config_features.yaml")
    print(f"  Configurations: {len(result['configurations'])}")
    print(f"  Written: {args.output}")


if __name__ == "__main__":
    main()
