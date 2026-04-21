#!/usr/bin/env python3
"""
Pipeline step 1 (paper §5): extract query-level features to YAML.

- Static features (Phi_static): concrete numbers from code structure.
- Symbolic features (Phi_sym): workload templates with formulas; values are
  instantiated later using graph/partition statistics.

Output: query_features.yaml

Example:
  python experiments/scripts/step1_query_features.py \\
    -i path/to/query.py -o out/query_features.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root: experiments/scripts -> ../..
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from autoconfig.utils.query_feature_extractor import QueryFeatureExtractor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Query source file (e.g. .py or pseudo-code)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="out/query_features.yaml",
        help="Output YAML path",
    )
    args = parser.parse_args()

    extractor = QueryFeatureExtractor()
    features = extractor.extract_from_file(args.input)
    features["metadata"] = {
        "pipeline_step": 1,
        "input_file": str(args.input),
        "output_file": str(args.output),
    }
    extractor.save_to_yaml(features, args.output)

    fc = features["feature_count"]
    print("Step 1 complete: query_features.yaml")
    print(f"  Static: {fc['static']}")
    print(f"  Symbolic template families: {fc['symbolic_families']}")
    print(f"  Written: {args.output}")


if __name__ == "__main__":
    main()
