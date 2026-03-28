"""
Query Code Feature Extractor

Extracts features from graph query source code and outputs to YAML.

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
    - Symbolic features (workload patterns)
    """
    
    def __init__(self):
        self.static_extractor = StaticFeatureExtractor()
        self.symbolic_extractor = SymbolicFeatureExtractor()
    
    def extract(
        self,
        source_code: str,
        graph_stats: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Extract features from query code.
        
        Args:
            source_code: Query source code string
            graph_stats: Optional graph statistics for symbolic feature instantiation
            
        Returns:
            Dictionary of features
        """
        # Extract static features
        static_features = self.static_extractor.extract(source_code)
        static_names = self.static_extractor.get_feature_names()
        
        # Extract symbolic features
        if graph_stats is None:
            # Use default stats for template-only extraction
            graph_stats = {
                'num_vertices': 1000,
                'num_edges': 5000,
                'diameter': 5,
                'avg_degree': 10,
                'max_degree': 50,
                'skew': 2.0,
            }
        
        symbolic_features = self.symbolic_extractor.extract(
            source_code, graph_stats
        )
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
        filepath: str,
        graph_stats: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Extract features from a source file.
        
        Args:
            filepath: Path to source file
            graph_stats: Optional graph statistics
            
        Returns:
            Dictionary of features
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        return self.extract(source_code, graph_stats)
    
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
    parser.add_argument(
        '--num-vertices',
        type=int,
        default=1000,
        help='Estimated number of vertices (for symbolic features)'
    )
    parser.add_argument(
        '--num-edges',
        type=int,
        default=5000,
        help='Estimated number of edges (for symbolic features)'
    )
    parser.add_argument(
        '--diameter',
        type=int,
        default=5,
        help='Estimated graph diameter'
    )
    
    args = parser.parse_args()
    
    # Extract features
    extractor = QueryFeatureExtractor()
    
    # Optional graph stats
    graph_stats = {
        'num_vertices': args.num_vertices,
        'num_edges': args.num_edges,
        'diameter': args.diameter,
        'avg_degree': args.num_edges / max(args.num_vertices, 1),
        'max_degree': args.num_edges / args.num_vertices * 2,
        'skew': 2.0,
    }
    
    features = extractor.extract_from_file(args.input, graph_stats)
    
    # Add metadata
    features['metadata'] = {
        'input_file': str(args.input),
        'output_file': args.output,
    }
    
    # Save to YAML
    extractor.save_to_yaml(features, args.output)
    
    print(f"Query features extracted:")
    print(f"  Static features: {features['feature_count']['static']}")
    print(f"  Symbolic features: {features['feature_count']['symbolic']}")
    print(f"  Total: {features['feature_count']['total']}")
    print(f"  Output: {args.output}")


if __name__ == '__main__':
    main()
