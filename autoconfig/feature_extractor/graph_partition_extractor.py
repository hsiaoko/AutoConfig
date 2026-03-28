"""
Graph and Partition Feature Extractor
Extracts statistics from input graph G and partitioning F.

Used to instantiate symbolic features from symbolic_extractor.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx


class GraphPartitionExtractor:
    """
    Extracts graph and partition statistics.
    
    These statistics are used to instantiate symbolic features:
    - |V|: Number of vertices
    - |E|: Number of edges
    - D: Graph diameter
    - avg_degree: Average degree
    - max_degree: Maximum degree
    - skew: Degree skew (max_degree / avg_degree)
    - boundary_degree_sum: Sum of boundary vertex degrees
    - num_partitions: Number of partitions
    """
    
    def __init__(self):
        self.feature_names = [
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
        ]
    
    def extract(
        self,
        graph: nx.Graph,
        partitions: Optional[Dict[Any, List[Any]]] = None
    ) -> np.ndarray:
        """
        Extract graph and partition statistics.
        
        Args:
            graph: NetworkX graph
            partitions: Dict mapping partition_id -> list of vertex_ids
            
        Returns:
            numpy array of graph and partition statistics
        """
        features = []
        
        # Graph statistics
        num_vertices = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        
        features.append(float(num_vertices))
        features.append(float(num_edges))
        
        # Diameter (sample for large graphs)
        diameter = self._compute_diameter(graph)
        features.append(float(diameter))
        
        # Degree statistics
        degrees = [d for n, d in graph.degree()]
        if degrees:
            avg_degree = np.mean(degrees)
            max_degree = np.max(degrees)
            min_degree = np.min(degrees)
            degree_std = np.std(degrees) if len(degrees) > 1 else 0.0
        else:
            avg_degree = max_degree = min_degree = degree_std = 0.0
        
        features.extend([
            float(avg_degree),
            float(max_degree),
            float(min_degree),
            float(degree_std),
        ])
        
        # Skew
        skew = max_degree / max(avg_degree, 1e-8)
        features.append(float(skew))
        
        # Clustering coefficient
        clustering = nx.average_clustering(graph)
        features.append(float(clustering))
        
        # Number of components
        num_components = nx.number_connected_components(graph)
        features.append(float(num_components))
        
        # Partition statistics
        if partitions:
            partition_features = self._extract_partition_features(
                graph, partitions
            )
        else:
            partition_features = self._default_partition_features()
        
        features.extend(partition_features)
        
        return np.array(features, dtype=np.float64)
    
    def _compute_diameter(self, graph: nx.Graph) -> int:
        """Compute graph diameter with sampling for large graphs."""
        if graph.number_of_nodes() == 0:
            return 0
        
        if not nx.is_connected(graph):
            # Use largest component
            components = nx.connected_components(graph)
            largest_component = max(components, key=len)
            subgraph = graph.subgraph(largest_component)
        else:
            subgraph = graph
        
        # Sample for large graphs
        if subgraph.number_of_nodes() > 1000:
            sample_size = min(100, subgraph.number_of_nodes())
            sample_nodes = list(np.random.choice(
                list(subgraph.nodes()), sample_size, replace=False
            ))
            sample_subgraph = subgraph.subgraph(sample_nodes)
            
            try:
                return nx.diameter(sample_subgraph)
            except:
                return 0
        
        try:
            return nx.diameter(subgraph)
        except:
            return 0
    
    def _extract_partition_features(
        self,
        graph: nx.Graph,
        partitions: Dict[Any, List[Any]]
    ) -> List[float]:
        """Extract partition-specific features."""
        features = []
        
        num_partitions = len(partitions)
        features.append(float(num_partitions))
        
        # Boundary vertices (vertices with neighbors in other partitions)
        boundary_vertices = set()
        boundary_degree_sum = 0
        
        # Create vertex -> partition mapping
        vertex_to_partition = {}
        for partition_id, vertices in partitions.items():
            for v in vertices:
                vertex_to_partition[v] = partition_id
        
        # Find boundary vertices
        for v in graph.nodes():
            v_partition = vertex_to_partition.get(v)
            if v_partition is None:
                continue
            
            for neighbor in graph.neighbors(v):
                n_partition = vertex_to_partition.get(neighbor)
                if n_partition is not None and n_partition != v_partition:
                    boundary_vertices.add(v)
                    boundary_degree_sum += 1
                    break
        
        features.append(float(len(boundary_vertices)))
        features.append(float(boundary_degree_sum))
        
        # Partition size statistics
        partition_sizes = [len(vertices) for vertices in partitions.values()]
        avg_size = np.mean(partition_sizes) if partition_sizes else 0.0
        size_std = np.std(partition_sizes) if len(partition_sizes) > 1 else 0.0
        
        features.append(float(avg_size))
        features.append(float(size_std))
        
        # Edge cut ratio
        total_edges = graph.number_of_edges()
        cut_edges = 0
        
        for u, v in graph.edges():
            u_partition = vertex_to_partition.get(u)
            v_partition = vertex_to_partition.get(v)
            if u_partition is not None and v_partition is not None:
                if u_partition != v_partition:
                    cut_edges += 1
        
        edge_cut_ratio = cut_edges / max(total_edges, 1)
        features.append(float(edge_cut_ratio))
        
        # Partition balance (1.0 = perfectly balanced)
        if partition_sizes and avg_size > 0:
            max_size = np.max(partition_sizes)
            balance = avg_size / max_size
        else:
            balance = 1.0
        
        features.append(float(balance))
        
        return features
    
    def _default_partition_features(self) -> List[float]:
        """Return default partition features when no partition info available."""
        return [
            1.0,   # num_partitions (assume single partition)
            0.0,   # boundary_vertices
            0.0,   # boundary_degree_sum
            0.0,   # avg_partition_size
            0.0,   # size_std
            0.0,   # edge_cut_ratio
            1.0,   # balance
        ]
    
    def extract_from_dict(
        self,
        graph_dict: Dict[str, Any],
        partitions: Optional[Dict[Any, List[Any]]] = None
    ) -> np.ndarray:
        """
        Extract from dictionary representation.
        
        Args:
            graph_dict: Dict with 'nodes' and 'edges' keys
            partitions: Optional partition mapping
            
        Returns:
            numpy array of features
        """
        nodes = graph_dict.get('nodes', [])
        edges = graph_dict.get('edges', [])
        
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        
        return self.extract(graph, partitions)
    
    def get_graph_stats_dict(
        self,
        graph: nx.Graph,
        partitions: Optional[Dict[Any, List[Any]]] = None
    ) -> Dict[str, Any]:
        """
        Get graph statistics as dictionary for symbolic feature instantiation.
        
        Args:
            graph: NetworkX graph
            partitions: Optional partition mapping
            
        Returns:
            Dictionary with graph statistics
        """
        features = self.extract(graph, partitions)
        
        return {
            'num_vertices': int(features[0]),
            'num_edges': int(features[1]),
            'diameter': int(features[2]),
            'avg_degree': float(features[3]),
            'max_degree': float(features[4]),
            'min_degree': float(features[5]),
            'degree_std': float(features[6]),
            'skew': float(features[7]),
            'clustering_coeff': float(features[8]),
            'num_components': int(features[9]),
            'num_partitions': int(features[10]),
            'boundary_vertices': int(features[11]),
            'boundary_degree_sum': float(features[12]),
            'avg_partition_size': float(features[13]),
            'partition_size_std': float(features[14]),
            'edge_cut_ratio': float(features[15]),
            'balance': float(features[16]),
        }
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names.copy()


class PartitionQualityMetrics:
    """
    Computes partition quality metrics.
    
    These metrics help understand the impact of partitioning on performance.
    """
    
    @staticmethod
    def edge_cut_ratio(graph: nx.Graph, partitions: Dict[Any, List[Any]]) -> float:
        """Compute edge cut ratio."""
        total_edges = graph.number_of_edges()
        if total_edges == 0:
            return 0.0
        
        # Create vertex -> partition mapping
        vertex_to_partition = {}
        for partition_id, vertices in partitions.items():
            for v in vertices:
                vertex_to_partition[v] = partition_id
        
        # Count cut edges
        cut_edges = 0
        for u, v in graph.edges():
            u_part = vertex_to_partition.get(u)
            v_part = vertex_to_partition.get(v)
            if u_part is not None and v_part is not None and u_part != v_part:
                cut_edges += 1
        
        return cut_edges / total_edges
    
    @staticmethod
    def balance_score(partitions: Dict[Any, List[Any]]) -> float:
        """
        Compute partition balance score.
        
        Returns:
            1.0 = perfectly balanced, 0.0 = completely unbalanced
        """
        if not partitions:
            return 1.0
        
        sizes = [len(vertices) for vertices in partitions.values()]
        avg_size = np.mean(sizes)
        max_size = np.max(sizes)
        
        if max_size == 0:
            return 1.0
        
        return avg_size / max_size
    
    @staticmethod
    def boundary_ratio(
        graph: nx.Graph,
        partitions: Dict[Any, List[Any]]
    ) -> float:
        """Compute ratio of boundary vertices."""
        total_vertices = graph.number_of_nodes()
        if total_vertices == 0:
            return 0.0
        
        # Create vertex -> partition mapping
        vertex_to_partition = {}
        for partition_id, vertices in partitions.items():
            for v in vertices:
                vertex_to_partition[v] = partition_id
        
        # Find boundary vertices
        boundary_vertices = set()
        for v in graph.nodes():
            v_partition = vertex_to_partition.get(v)
            if v_partition is None:
                continue
            
            for neighbor in graph.neighbors(v):
                n_partition = vertex_to_partition.get(neighbor)
                if n_partition is not None and n_partition != v_partition:
                    boundary_vertices.add(v)
                    break
        
        return len(boundary_vertices) / total_vertices
    
    @staticmethod
    def comprehensive_score(
        graph: nx.Graph,
        partitions: Dict[Any, List[Any]]
    ) -> float:
        """
        Compute comprehensive partition quality score.
        
        Lower is better (combines edge cut, imbalance, boundary ratio).
        """
        edge_cut = PartitionQualityMetrics.edge_cut_ratio(graph, partitions)
        balance = 1.0 - PartitionQualityMetrics.balance_score(partitions)
        boundary = PartitionQualityMetrics.boundary_ratio(graph, partitions)
        
        # Weighted combination
        return 0.5 * edge_cut + 0.3 * balance + 0.2 * boundary
