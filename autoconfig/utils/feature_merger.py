"""
Feature Merge Tool

Merges query, graph, and configuration features into a single feature vector.
Instantiates symbolic features with actual graph statistics.

Usage:
    python -m autoconfig.utils.feature_merger \
        --query out/query_features.yaml \
        --graph out/graph_features.yaml \
        --config out/config_features.yaml \
        --output out/merged_features.yaml
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple


class FeatureMerger:
    """
    Merges features from query, graph, and configuration.
    
    Process:
    1. Load query features (with placeholders)
    2. Load graph features (with actual statistics)
    3. Load config features (multiple configurations)
    4. Instantiate symbolic features with graph stats
    5. Merge into complete feature vectors
    """
    
    def __init__(self):
        self.feature_order = [
            # Static features (8)
            'static_loop_count',
            'static_max_loop_depth',
            'static_branch_count',
            'static_variable_count',
            'static_recursion_count',
            'static_atomic_op_count',
            'static_sync_count',
            'static_explicit_parallel_flag',
            # Symbolic features (12)
            'sym_vscan_coeff',
            'sym_vscan_requires',
            'sym_escan_coeff',
            'sym_escan_requires',
            'sym_fscan_coeff',
            'sym_fscan_requires',
            'sym_rexp_coeff',
            'sym_rexp_requires',
            'sym_atom_coeff',
            'sym_atom_requires',
            'sym_comm_coeff',
            'sym_comm_requires',
            # Graph features (17)
            'graph_num_vertices',
            'graph_num_edges',
            'graph_diameter',
            'graph_avg_degree',
            'graph_max_degree',
            'graph_min_degree',
            'graph_degree_std',
            'graph_skew',
            'graph_clustering_coeff',
            'graph_num_components',
            'partition_num_partitions',
            'partition_boundary_vertices',
            'partition_boundary_degree_sum',
            'partition_avg_partition_size',
            'partition_size_std',
            'partition_edge_cut_ratio',
            'partition_balance',
            # Config features (10)
            'conf_memory_limit',
            'conf_num_threads',
            'conf_cache_size',
            'conf_batch_size',
            'conf_io_buffer_size',
            'conf_num_workers',
            'conf_timeout',
            'conf_enable_index',
            'conf_index_type',
            'conf_compression_enabled',
        ]
    
    def load_yaml(self, filepath: str) -> Dict[str, Any]:
        """Load YAML file (handling numpy types)."""
        with open(filepath, 'r', encoding='utf-8') as f:
            # Use unsafe loader to handle numpy types
            return yaml.unsafe_load(f)
    
    def instantiate_symbolic(
        self,
        symbolic_features: Dict[str, float],
        graph_stats: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Instantiate symbolic features with graph statistics.
        
        Args:
            symbolic_features: Symbolic features with placeholders
            graph_stats: Graph statistics for instantiation
            
        Returns:
            Instantiated symbolic features
        """
        instantiated = symbolic_features.copy()
        
        # VScan: |V| or |V|/n
        if symbolic_features.get('sym_vscan_requires', 0) > 0:
            instantiated['sym_vscan_coeff'] = graph_stats.get('num_vertices', 0)
        
        # EScan: |E| or |E|/n
        if symbolic_features.get('sym_escan_requires', 0) > 0:
            instantiated['sym_escan_coeff'] = graph_stats.get('num_edges', 0)
        
        # FScan: D × |E|/D = |E| (approximation)
        if symbolic_features.get('sym_fscan_requires', 0) > 0:
            diameter = graph_stats.get('diameter', 1)
            num_edges = graph_stats.get('num_edges', 0)
            instantiated['sym_fscan_coeff'] = num_edges  # Simplified: D * (E/D)
        
        # RExp: avg_degree ^ diameter
        if symbolic_features.get('sym_rexp_requires', 0) > 0:
            avg_degree = graph_stats.get('avg_degree', 2)
            diameter = min(graph_stats.get('diameter', 3), 10)  # Cap to avoid overflow
            instantiated['sym_rexp_coeff'] = avg_degree ** diameter
        
        # Atom: |E| × skew
        if symbolic_features.get('sym_atom_requires', 0) > 0:
            num_edges = graph_stats.get('num_edges', 0)
            skew = graph_stats.get('skew', 1)
            instantiated['sym_atom_coeff'] = num_edges * skew
        
        # Comm: boundary_degree_sum
        if symbolic_features.get('sym_comm_requires', 0) > 0:
            instantiated['sym_comm_coeff'] = graph_stats.get('boundary_degree_sum', 0)
        
        return instantiated
    
    def extract_graph_features(
        self,
        graph_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract graph features from loaded graph YAML."""
        gf = graph_data.get('graph_features', {})
        
        return {
            'graph_num_vertices': gf.get('basic', {}).get('num_vertices', 0),
            'graph_num_edges': gf.get('basic', {}).get('num_edges', 0),
            'graph_diameter': gf.get('structure', {}).get('diameter', 0),
            'graph_avg_degree': gf.get('degree', {}).get('avg', 0),
            'graph_max_degree': gf.get('degree', {}).get('max', 0),
            'graph_min_degree': gf.get('degree', {}).get('min', 0),
            'graph_degree_std': gf.get('degree', {}).get('std', 0),
            'graph_skew': gf.get('degree', {}).get('skew', 0),
            'graph_clustering_coeff': gf.get('structure', {}).get('clustering_coeff', 0),
            'graph_num_components': gf.get('structure', {}).get('num_components', 1),
            'partition_num_partitions': gf.get('partition', {}).get('num_partitions', 1),
            'partition_boundary_vertices': gf.get('partition', {}).get('boundary_vertices', 0),
            'partition_boundary_degree_sum': gf.get('partition', {}).get('boundary_degree_sum', 0),
            'partition_avg_partition_size': gf.get('partition', {}).get('avg_partition_size', 0),
            'partition_size_std': gf.get('partition', {}).get('partition_size_std', 0),
            'partition_edge_cut_ratio': gf.get('partition', {}).get('edge_cut_ratio', 0),
            'partition_balance': gf.get('partition', {}).get('balance', 1),
        }
    
    def extract_config_features(
        self,
        config_data: Dict[str, Any]
    ) -> List[Dict[str, float]]:
        """Extract config features from loaded config YAML."""
        config_features = config_data.get('config_features', [])
        
        if not config_features:
            # Single configuration
            return [self._config_to_features(config_data.get('configurations', [{}])[0])]
        
        return [self._config_to_features(cf) for cf in config_features]
    
    def _config_to_features(
        self,
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Convert single config to feature dictionary."""
        k = config.get('k', 1)
        resource = config.get('resource', config.get('conf_per_instance', {}))
        
        return {
            'conf_memory_limit': k * resource.get('memory_gb', 8) * 1024,  # Convert to MB
            'conf_num_threads': k * resource.get('cpu_cores', 4),
            'conf_cache_size': k * resource.get('memory_gb', 8) * 1024 // 8,  # 1/8 of memory
            'conf_batch_size': 1000,  # Default
            'conf_io_buffer_size': 64,  # Default
            'conf_num_workers': k * max(1, resource.get('cpu_cores', 4) // 4),
            'conf_timeout': 300,  # Default
            'conf_enable_index': 1.0,  # Default enabled
            'conf_index_type': 1.0,  # Default btree
            'conf_compression_enabled': 0.0,  # Default disabled
        }
    
    def merge(
        self,
        query_features: Dict[str, Any],
        graph_features: Dict[str, Any],
        config_features: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Merge all features into feature vectors.
        
        Args:
            query_features: Query feature dictionary
            graph_features: Graph feature dictionary
            config_features: List of config feature dictionaries
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        # Extract static features
        static = query_features.get('query_features', {}).get('static', {})
        
        # Extract and instantiate symbolic features
        symbolic = query_features.get('query_features', {}).get('symbolic', {})
        graph_stats = self.extract_graph_features(graph_features)
        symbolic_instantiated = self.instantiate_symbolic(symbolic, graph_stats)
        
        # Build merged features for each configuration
        feature_vectors = []
        
        for cf in config_features:
            merged = {}
            merged.update(static)
            merged.update(symbolic_instantiated)
            merged.update(graph_stats)
            merged.update(cf)
            
            # Create feature vector in correct order
            vector = []
            for name in self.feature_order:
                vector.append(float(merged.get(name, 0.0)))
            
            feature_vectors.append(vector)
        
        return np.array(feature_vectors, dtype=np.float64), self.feature_order.copy()
    
    def merge_all(
        self,
        query_file: str,
        graph_file: str,
        config_file: str
    ) -> Dict[str, Any]:
        """
        Load and merge all features from files.
        
        Args:
            query_file: Path to query features YAML
            graph_file: Path to graph features YAML
            config_file: Path to config features YAML
            
        Returns:
            Complete merged feature data
        """
        # Load all files
        query_data = self.load_yaml(query_file)
        graph_data = self.load_yaml(graph_file)
        config_data = self.load_yaml(config_file)
        
        # Extract features
        config_features = self.extract_config_features(config_data)
        graph_stats = self.extract_graph_features(graph_data)
        
        # Merge
        feature_matrix, feature_names = self.merge(
            query_data,
            graph_data,
            config_features
        )
        
        # Build result
        result = {
            'feature_matrix': feature_matrix.tolist(),
            'feature_names': feature_names,
            'metadata': {
                'num_samples': len(config_features),
                'num_features': len(feature_names),
                'query_file': str(query_file),
                'graph_file': str(graph_file),
                'config_file': str(config_file),
            },
            'feature_groups': {
                'static': 8,
                'symbolic': 12,
                'graph': 17,
                'config': 10,
            }
        }
        
        return result
    
    def save_to_yaml(
        self,
        data: Dict[str, Any],
        output_path: str
    ):
        """Save merged features to YAML."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description='Merge query, graph, and configuration features'
    )
    parser.add_argument(
        '--query', '-q',
        type=str,
        required=True,
        help='Query features YAML file'
    )
    parser.add_argument(
        '--graph', '-g',
        type=str,
        required=True,
        help='Graph features YAML file'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Configuration features YAML file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='out/merged_features.yaml',
        help='Output merged features YAML file'
    )
    
    args = parser.parse_args()
    
    # Merge features
    merger = FeatureMerger()
    result = merger.merge_all(args.query, args.graph, args.config)
    
    # Save to YAML
    merger.save_to_yaml(result, args.output)
    
    # Print summary
    print(f"Features merged successfully:")
    print(f"  Query: {args.query}")
    print(f"  Graph: {args.graph}")
    print(f"  Config: {args.config}")
    print(f"  Output: {args.output}")
    print(f"  Feature matrix: {result['metadata']['num_samples']} samples × {result['metadata']['num_features']} features")
    print(f"\nFeature groups:")
    for group, count in result['feature_groups'].items():
        print(f"  {group}: {count}")


if __name__ == '__main__':
    main()
