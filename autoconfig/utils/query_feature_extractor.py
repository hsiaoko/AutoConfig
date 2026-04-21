"""
Query Code Feature Extractor

Extracts features from graph query source code and outputs to YAML.
Static features are numeric; symbolic features are template names + formulas
(see SymbolicFeatureExtractor.extract_symbolic_expressions).

Usage:
    python -m autoconfig.utils.query_feature_extractor \
        --input query.py \
        --output out/query_features.yaml
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any

from ..feature_extractor import StaticFeatureExtractor, SymbolicFeatureExtractor


class QueryFeatureExtractor:
    """
    Extract features from graph query source code.

    Outputs:
    - Static features (code structure)
    - Symbolic features (workload patterns with placeholders)
    """

    def __init__(self):
        self.static_extractor = StaticFeatureExtractor()
        self.symbolic_extractor = SymbolicFeatureExtractor()

    def extract(
        self,
        source_code: str
    ) -> Dict[str, Any]:
        """
        Extract features from query code.

        Symbolic features are templates (names + formulas); numeric instantiation
        happens in the merge step with graph_features.yaml.

        Args:
            source_code: Query source code string
            
        Returns:
            Dictionary of features with placeholders
        """
        # Extract static features (concrete scalars from code structure)
        static_features = self.static_extractor.extract(source_code)
        static_names = self.static_extractor.get_feature_names()

        # Symbolic: template names + formulas; values materialize at merge with graph YAML
        symbolic_payload = self.symbolic_extractor.extract_symbolic_expressions(source_code)

        # Build result dictionary
        result = {
            'query_features': {
                'static': self._features_to_dict(static_features, static_names),
                'symbolic': symbolic_payload,
            },
            'feature_count': {
                'static': len(static_features),
                'symbolic_families': len(symbolic_payload['families']),
                'symbolic_numeric_dim_after_merge': len(self.symbolic_extractor.get_feature_names()),
                'total_static_plus_symbolic_slots': len(static_features) + len(
                    self.symbolic_extractor.get_feature_names()
                ),
            },
            'placeholders': {
                'note': (
                    'Symbolic entries are partial functions (template + formula). '
                    'Step 4 merges with graph_features.yaml and system config to produce numeric vectors.'
                ),
            }
        }

        return result

    def _features_to_dict(
        self,
        features: np.ndarray,
        names: list
    ) -> Dict[str, float]:
        """Convert feature array to dictionary."""
        return {name: float(value) for name, value in zip(names, features)}

    def extract_from_file(
        self,
        filepath: str
    ) -> Dict[str, Any]:
        """
        Extract features from a source file.

        Args:
            filepath: Path to source file

        Returns:
            Dictionary of features
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            source_code = f.read()

        return self.extract(source_code)

    def save_to_yaml(
        self,
        features: Dict[str, Any],
        output_path: str
    ):
        """
        Save features to YAML file.

        Args:
            features: Feature dictionary
            output_path: Output file path
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, 'w', encoding='utf-8') as f:
            yaml.dump(features, f, default_flow_style=False, allow_unicode=True)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description='Extract features from graph query source code'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input query source file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='out/query_features.yaml',
        help='Output YAML file path'
    )

    args = parser.parse_args()

    # Extract features
    extractor = QueryFeatureExtractor()
    features = extractor.extract_from_file(args.input)

    # Add metadata
    features['metadata'] = {
        'input_file': str(args.input),
        'output_file': args.output,
    }

    # Save to YAML
    extractor.save_to_yaml(features, args.output)

    fc = features['feature_count']
    print("Query features extracted:")
    print(f"  Static scalars: {fc['static']}")
    print(f"  Symbolic families (templates / formulas): {fc['symbolic_families']}")
    print(f"  After merge, symbolic becomes {fc['symbolic_numeric_dim_after_merge']} numeric features")
    print(f"  Output: {args.output}")


if __name__ == '__main__':
    main()
