#!/usr/bin/env python3
"""
Pipeline step 4: merge query_features + graph_features + system config into
the final numeric feature matrix (YAML).

Symbolic templates from step 1 are instantiated here using graph statistics
from step 2; one row per configuration sample from step 3.

Example:
  python experiments/scripts/step4_merge_features.py \\
    -q out/query_features.yaml \\
    -g out/graph_features.yaml \\
    -c out/config_features.yaml \\
    -o out/merged_features.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from autoconfig.utils.feature_merger import FeatureMerger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", "-q", required=True, help="query_features.yaml from step 1")
    parser.add_argument("--graph", "-g", required=True, help="graph_features.yaml from step 2")
    parser.add_argument("--config", "-c", required=True, help="config_features.yaml from step 3")
    parser.add_argument(
        "--output",
        "-o",
        default="out/merged_features.yaml",
        help="Merged features YAML",
    )
    args = parser.parse_args()

    merger = FeatureMerger()
    result = merger.merge_all(args.query, args.graph, args.config)
    result.setdefault("metadata", {})["pipeline_step"] = 4
    merger.save_to_yaml(result, args.output)

    m = result["metadata"]
    print("Step 4 complete: merged numeric features")
    print(f"  Matrix: {m['num_samples']} x {m['num_features']}")
    print(f"  Written: {args.output}")


if __name__ == "__main__":
    main()
