"""
Query Feature Extractor
Extracts features from graph query patterns (Q).
"""

import numpy as np
from typing import Dict, Any, Optional, List
import networkx as nx


class QueryFeatureExtractor:
    """
    Extracts features from a graph query pattern.
    
    Query features include:
    - Number of vertices and edges in the query pattern
    - Query graph density
    - Degree statistics
    - Structural properties (cycles, connectivity, etc.)
    """
    
    def __init__(self):
        self.feature_names = [
            'query_num_nodes',
            'query_num_edges',
            'query_density',
            'query_avg_degree',
            'query_max_degree',
            'query_min_degree',
            'query_std_degree',
            'query_num_triangles',
            'query_clustering_coeff',
            'query_diameter',
            'query_is_connected',
            'query_num_components',
            'query_avg_path_length',
        ]
    
    def extract(self, query_graph: nx.Graph) -> np.ndarray:
        """
        Extract features from a query graph.
        
        Args:
            query_graph: NetworkX graph representing the query pattern
            
        Returns:
            numpy array of query features
        """
        features = []
        
        # Basic statistics
        num_nodes = query_graph.number_of_nodes()
        num_edges = query_graph.number_of_edges()
        features.append(num_nodes)
        features.append(num_edges)
        
        # Density
        density = nx.density(query_graph)
        features.append(density)
        
        # Degree statistics
        degrees = [d for n, d in query_graph.degree()]
        if degrees:
            features.append(np.mean(degrees))
            features.append(np.max(degrees))
            features.append(np.min(degrees))
            features.append(np.std(degrees) if len(degrees) > 1 else 0.0)
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Triangle count
        num_triangles = sum(nx.triangles(query_graph).values()) // 3
        features.append(num_triangles)
        
        # Clustering coefficient
        clustering_coeff = nx.average_clustering(query_graph)
        features.append(clustering_coeff)
        
        # Diameter (only for connected graphs)
        if nx.is_connected(query_graph):
            try:
                diameter = nx.diameter(query_graph)
                features.append(diameter)
            except:
                features.append(0)
        else:
            features.append(0)
        
        # Connectivity
        is_connected = 1 if nx.is_connected(query_graph) else 0
        features.append(is_connected)
        
        num_components = nx.number_connected_components(query_graph)
        features.append(num_components)
        
        # Average path length
        if nx.is_connected(query_graph):
            try:
                avg_path_length = nx.average_shortest_path_length(query_graph)
                features.append(avg_path_length)
            except:
                features.append(0.0)
        else:
            features.append(0.0)
        
        return np.array(features, dtype=np.float64)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names.copy()
    
    def extract_from_dict(self, query_dict: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from a query dictionary representation.
        
        Args:
            query_dict: Dictionary with 'nodes' and 'edges' keys
                       e.g., {'nodes': [0,1,2], 'edges': [(0,1), (1,2)]}
                       
        Returns:
            numpy array of query features
        """
        nodes = query_dict.get('nodes', [])
        edges = query_dict.get('edges', [])
        
        query_graph = nx.Graph()
        query_graph.add_nodes_from(nodes)
        query_graph.add_edges_from(edges)
        
        return self.extract(query_graph)
