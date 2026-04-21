#!/usr/bin/env python3
"""
Pipeline step 2: graph data + partition statistics from an edge list (or folder).

Output: graph_features.yaml (used to instantiate symbolic templates in step 4).

Example:
  python experiments/scripts/step2_graph_features.py \\
    -i data/edges.csv -o out/graph_features.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from autoconfig.utils.graph_feature_extractor import GraphFeatureExtractor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Edge list file or directory of partition edge files",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="out/graph_features.yaml",
        help="Output YAML path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    extractor = GraphFeatureExtractor()

    if input_path.is_file():
        features = extractor.extract_single(str(input_path))
    elif input_path.is_dir():
        features = extractor.extract_partitioned(str(input_path))
    else:
        print(f"Error: not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    features["metadata"] = {
        "pipeline_step": 2,
        **features.get("metadata", {}),
        "output_file": str(args.output),
    }
    extractor.save_to_yaml(features, args.output)

    gf = features["graph_features"]
    print("Step 2 complete: graph_features.yaml")
    print(f"  |V|={gf['basic']['num_vertices']}, |E|={gf['basic']['num_edges']}")
    print(f"  Written: {args.output}")


if __name__ == "__main__":
    main()
