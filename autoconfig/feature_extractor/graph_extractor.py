"""
Graph Feature Extractor
Extracts features from the data graph (G).
"""

import numpy as np
from typing import Dict, Any, List
import networkx as nx


class GraphFeatureExtractor:
    """
    Extracts features from the data graph.
    
    Graph features include:
    - Scale statistics (nodes, edges)
    - Structural properties (density, degree distribution)
    - Connectivity patterns
    - Community structure
    """
    
    def __init__(self):
        self.feature_names = [
            'graph_num_nodes',
            'graph_num_edges',
            'graph_density',
            'graph_avg_degree',
            'graph_max_degree',
            'graph_min_degree',
            'graph_std_degree',
            'graph_num_triangles',
            'graph_clustering_coeff',
            'graph_diameter',
            'graph_is_connected',
            'graph_num_components',
            'graph_avg_path_length',
            'graph_assortativity',
            'graph_transitivity',
        ]
    
    def extract(self, data_graph: nx.Graph) -> np.ndarray:
        """
        Extract features from a data graph.
        
        Args:
            data_graph: NetworkX graph representing the data graph
            
        Returns:
            numpy array of graph features
        """
        features = []
        
        # Basic statistics
        num_nodes = data_graph.number_of_nodes()
        num_edges = data_graph.number_of_edges()
        features.append(float(num_nodes))
        features.append(float(num_edges))
        
        # Density
        density = nx.density(data_graph)
        features.append(density)
        
        # Degree statistics
        degrees = [d for n, d in data_graph.degree()]
        if degrees:
            features.append(np.mean(degrees))
            features.append(np.max(degrees))
            features.append(np.min(degrees))
            features.append(np.std(degrees) if len(degrees) > 1 else 0.0)
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Triangle count
        num_triangles = sum(nx.triangles(data_graph).values()) // 3
        features.append(float(num_triangles))
        
        # Clustering coefficient
        clustering_coeff = nx.average_clustering(data_graph)
        features.append(clustering_coeff)
        
        # Diameter (sample for large graphs)
        if nx.is_connected(data_graph):
            try:
                if num_nodes > 1000:
                    # Sample for large graphs
                    sample_nodes = list(np.random.choice(
                        list(data_graph.nodes()), 
                        min(100, num_nodes), 
                        replace=False
                    ))
                    subgraph = data_graph.subgraph(sample_nodes)
                    diameter = nx.diameter(subgraph) if nx.is_connected(subgraph) else 0
                else:
                    diameter = nx.diameter(data_graph)
                features.append(float(diameter))
            except:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Connectivity
        is_connected = 1.0 if nx.is_connected(data_graph) else 0.0
        features.append(is_connected)
        
        num_components = nx.number_connected_components(data_graph)
        features.append(float(num_components))
        
        # Average path length (sample for large graphs)
        if nx.is_connected(data_graph):
            try:
                if num_nodes > 1000:
                    sample_nodes = list(np.random.choice(
                        list(data_graph.nodes()),
                        min(100, num_nodes),
                        replace=False
                    ))
                    subgraph = data_graph.subgraph(sample_nodes)
                    avg_path = nx.average_shortest_path_length(subgraph) if nx.is_connected(subgraph) else 0.0
                else:
                    avg_path = nx.average_shortest_path_length(data_graph)
                features.append(avg_path)
            except:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Assortativity
        try:
            assortativity = nx.degree_assortativity_coefficient(data_graph)
            features.append(assortativity)
        except:
            features.append(0.0)
        
        # Transitivity
        transitivity = nx.transitivity(data_graph)
        features.append(transitivity)
        
        return np.array(features, dtype=np.float64)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names.copy()
    
    def extract_from_dict(self, graph_dict: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from a graph dictionary representation.
        
        Args:
            graph_dict: Dictionary with 'nodes' and 'edges' keys
            
        Returns:
            numpy array of graph features
        """
        nodes = graph_dict.get('nodes', [])
        edges = graph_dict.get('edges', [])
        
        data_graph = nx.Graph()
        data_graph.add_nodes_from(nodes)
        data_graph.add_edges_from(edges)
        
        return self.extract(data_graph)
