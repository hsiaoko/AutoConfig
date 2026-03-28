"""
Feature Manager
Combines static, symbolic, and graph/partition features.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import networkx as nx

from .static_extractor import StaticFeatureExtractor, ASTBasedStaticExtractor
from .symbolic_extractor import SymbolicFeatureExtractor
from .graph_partition_extractor import GraphPartitionExtractor


class FeatureManager:
    """
    Manages three-stage feature extraction for Hybrid tasks.
    
    Pipeline:
    1. Static feature extraction from query code
    2. Symbolic template matching
    3. Graph-aware instantiation
    """
    
    def __init__(self, use_ast_for_static: bool = True):
        """
        Initialize feature manager.
        
        Args:
            use_ast_for_static: Use AST-based extraction for Python code
        """
        self.static_extractor = (
            ASTBasedStaticExtractor() if use_ast_for_static
            else StaticFeatureExtractor()
        )
        self.symbolic_extractor = SymbolicFeatureExtractor()
        self.graph_extractor = GraphPartitionExtractor()
        
        # Combined feature names
        self.feature_names = (
            self.static_extractor.get_feature_names() +
            self.symbolic_extractor.get_feature_names() +
            self.graph_extractor.get_feature_names()
        )
    
    def extract_all(
        self,
        source_code: str,
        graph: nx.Graph,
        config: Dict[str, Any],
        partitions: Optional[Dict[Any, List[Any]]] = None
    ) -> np.ndarray:
        """
        Extract all features from code, graph, and config.
        
        Args:
            source_code: Query source code string
            graph: NetworkX graph representing input graph G
            config: Configuration dictionary
            partitions: Optional partition mapping
            
        Returns:
            Combined feature vector
        """
        # Stage 1: Static features from code
        static_features = self.static_extractor.extract(source_code)
        
        # Stage 2 & 3: Symbolic features with graph instantiation
        graph_stats = self.graph_extractor.get_graph_stats_dict(graph, partitions)
        partition_stats = graph_stats  # Use same dict for now
        
        symbolic_features = self.symbolic_extractor.extract(
            source_code, graph_stats, partition_stats
        )
        
        # Graph/partition statistics
        graph_features = self.graph_extractor.extract(graph, partitions)
        
        # Configuration features
        config_extractor = ConfigFeatureExtractor()
        config_features = config_extractor.extract(config)
        
        # Combine all features
        combined = np.concatenate([
            static_features,
            symbolic_features,
            graph_features,
            config_features
        ])
        
        return combined
    
    def extract_from_files(
        self,
        query_file: str,
        graph_file: str,
        config: Dict[str, Any],
        partition_file: Optional[str] = None
    ) -> np.ndarray:
        """
        Extract features from files.
        
        Args:
            query_file: Path to query source file
            graph_file: Path to graph file (GML, GraphML, etc.)
            config: Configuration dictionary
            partition_file: Optional path to partition file
            
        Returns:
            Combined feature vector
        """
        # Load source code
        with open(query_file, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Load graph
        if graph_file.endswith('.gml'):
            graph = nx.read_gml(graph_file)
        elif graph_file.endswith('.graphml'):
            graph = nx.read_graphml(graph_file)
        else:
            raise ValueError(f"Unsupported graph format: {graph_file}")
        
        # Load partitions if available
        partitions = None
        if partition_file:
            partitions = self._load_partitions(partition_file)
        
        return self.extract_all(source_code, graph, config, partitions)
    
    def _load_partitions(
        self,
        partition_file: str
    ) -> Optional[Dict[Any, List[Any]]]:
        """Load partition mapping from file."""
        import json
        
        with open(partition_file, 'r') as f:
            data = json.load(f)
        
        # Convert string keys to appropriate types
        partitions = {}
        for partition_id, vertices in data.items():
            partitions[int(partition_id)] = vertices
        
        return partitions
    
    def get_feature_names(self) -> List[str]:
        """Return all feature names."""
        return self.feature_names.copy()
    
    def get_feature_dimensions(self) -> Tuple[int, int, int, int]:
        """Return dimensions of each feature group."""
        return (
            len(self.static_extractor.get_feature_names()),
            len(self.symbolic_extractor.get_feature_names()),
            len(self.graph_extractor.get_feature_names()),
            len(ConfigFeatureExtractor().get_feature_names()),
        )
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Return feature names grouped by category."""
        return {
            'static': self.static_extractor.get_feature_names(),
            'symbolic': self.symbolic_extractor.get_feature_names(),
            'graph_partition': self.graph_extractor.get_feature_names(),
            'config': ConfigFeatureExtractor().get_feature_names(),
        }
    
    def normalize_features(
        self,
        features: np.ndarray,
        means: Optional[np.ndarray] = None,
        stds: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize features using z-score normalization.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            means: Pre-computed means
            stds: Pre-computed stds
            
        Returns:
            Tuple of (normalized, means, stds)
        """
        if means is None:
            means = np.mean(features, axis=0)
        if stds is None:
            stds = np.std(features, axis=0)
            stds[stds == 0] = 1.0
        
        normalized = (features - means) / stds
        return normalized, means, stds


# Import ConfigFeatureExtractor for use in FeatureManager
from .config_extractor import ConfigFeatureExtractor
