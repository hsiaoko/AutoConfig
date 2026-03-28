"""
Query Code Feature Extractor

Extracts features from graph query source code and outputs to YAML.
Uses placeholders for graph-dependent values.

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
        
        Graph-dependent symbolic features use placeholders that will be
        instantiated in the merge stage with actual graph statistics.

        Args:
            source_code: Query source code string
            
        Returns:
            Dictionary of features with placeholders
        """
        # Extract static features
        static_features = self.static_extractor.extract(source_code)
        static_names = self.static_extractor.get_feature_names()

        # Extract symbolic features (template-only, placeholders for graph stats)
        symbolic_features = self.symbolic_extractor.extract_templates(source_code)
        symbolic_names = self.symbolic_extractor.get_feature_names()

        # Build result dictionary
        result = {
            'query_features': {
                'static': self._features_to_dict(static_features, static_names),
                'symbolic': self._features_to_dict(symbolic_features, symbolic_names),
            },
            'feature_count': {
                'static': len(static_features),
                'symbolic': len(symbolic_features),
                'total': len(static_features) + len(symbolic_features),
            },
            'placeholders': {
                'note': 'Graph-dependent symbolic coefficients use placeholders. Will be instantiated in merge stage.',
                'required_graph_stats': [
                    'num_vertices', 'num_edges', 'diameter',
                    'avg_degree', 'max_degree', 'skew',
                    'boundary_degree_sum'
                ],
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

    print(f"Query features extracted:")
    print(f"  Static: {features['feature_count']['static']}")
    print(f"  Symbolic: {features['feature_count']['symbolic']}")
    print(f"  Total: {features['feature_count']['total']}")
    print(f"  Output: {args.output}")
    print(f"  Note: Graph-dependent values use placeholders")


if __name__ == '__main__':
    main()
