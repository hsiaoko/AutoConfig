"""
Graph Data Feature Extractor

Extracts features from graph data (edge list format) and outputs to YAML.
Supports both single graphs and partitioned graphs (folder input).

Usage:
    # Single graph
    python -m autoconfig.utils.graph_feature_extractor \
        --input data/edges.csv \
        --output out/graph_features.yaml
    
    # Partitioned graph (folder with multiple edge files)
    python -m autoconfig.utils.graph_feature_extractor \
        --input data/partitions/ \
        --output out/graph_features.yaml
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from collections import Counter

try:
    import networkx as nx
except ImportError:
    nx = None

from ..feature_extractor import GraphPartitionExtractor, PartitionQualityMetrics


class GraphFeatureExtractor:
    """
    Extract features from graph data.
    
    Supports:
    - Single graph (one edge list file)
    - Partitioned graph (folder with multiple edge list files)
    """
    
    # Above this size, loading the full graph into NetworkX is skipped; one streaming
    # pass computes exact |V|, |E|, degree stats, and components, and keeps a uniform
    # reservoir of edges for approximate diameter / clustering on a small NetworkX graph.
    _STREAMING_SIZE_THRESHOLD = 32 * 1024 * 1024
    
    def __init__(self):
        self.extractor = GraphPartitionExtractor()
    
    @staticmethod
    def _parse_edge_tokens(a: str, b: str) -> Tuple[Any, Any]:
        def tok(x: str) -> Union[int, str]:
            x = x.strip()
            try:
                return int(x)
            except ValueError:
                return x
        
        return tok(a), tok(b)
    
    def _parse_edge_line(self, line: str) -> Optional[Tuple[Any, Any]]:
        line = line.strip()
        if not line:
            return None
        if ',' in line:
            parts = line.split(',')
            if len(parts) >= 2:
                return self._parse_edge_tokens(parts[0], parts[1])
        parts = line.split()
        if len(parts) >= 2:
            return self._parse_edge_tokens(parts[0], parts[1])
        return None
    
    def load_edge_list(self, filepath: str) -> List[Tuple[Any, Any]]:
        """
        Load edges from an edge list file (comma-separated or whitespace-separated).
        
        Recognizes optional CSV headers: src,dst / source,target / from,to / src,tgt.
        """
        edges: List[Tuple[Any, Any]] = []
        is_first = True
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                if is_first:
                    is_first = False
                    if raw.lower() in ('src,dst', 'source,target', 'from,to', 'src,tgt'):
                        continue
                parsed = self._parse_edge_line(line)
                if parsed is None:
                    continue
                edges.append(parsed)
        
        return edges
    
    def build_graph(self, edges: List[Tuple[Any, Any]]) -> 'nx.Graph':
        """
        Build NetworkX graph from edge list.
        
        Args:
            edges: List of (source, destination) tuples
            
        Returns:
            NetworkX Graph
        """
        if nx is None:
            raise ImportError("networkx is required for graph feature extraction")
        
        graph = nx.Graph()
        
        for src, dst in edges:
            graph.add_edge(src, dst)
        
        return graph
    
    def _extract_single_streaming(self, edge_file: str) -> Dict[str, Any]:
        """Streaming extraction for edge lists too large for a full NetworkX graph."""
        if nx is None:
            raise ImportError("networkx is required for graph feature extraction")
        
        parent: Dict[Any, Any] = {}
        rank: Dict[Any, int] = {}
        
        def find(x: Any) -> Any:
            if x not in parent:
                parent[x] = x
                rank[x] = 0
                return x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(x: Any, y: Any) -> None:
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
        
        deg: Counter = Counter()
        n_edges = 0
        rng = np.random.default_rng(0)
        sample_edge_cap = 8000
        reservoir: List[Tuple[Any, Any]] = []
        
        with open(edge_file, 'r', encoding='utf-8') as f:
            is_first = True
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                if is_first:
                    is_first = False
                    if raw.lower() in ('src,dst', 'source,target', 'from,to', 'src,tgt'):
                        continue
                parsed = self._parse_edge_line(line)
                if parsed is None:
                    continue
                u, v = parsed
                union(u, v)
                deg[u] += 1
                deg[v] += 1
                n_edges += 1
                if len(reservoir) < sample_edge_cap:
                    reservoir.append((u, v))
                else:
                    j = int(rng.integers(0, n_edges))
                    if j < sample_edge_cap:
                        reservoir[j] = (u, v)
        
        n_vertices = len(deg)
        if n_vertices == 0:
            raise ValueError(f"No edges parsed from {edge_file}")
        
        deg_vals = list(deg.values())
        avg_degree = float(np.mean(deg_vals))
        max_degree = int(np.max(deg_vals))
        min_degree = int(np.min(deg_vals))
        degree_std = float(np.std(deg_vals)) if len(deg_vals) > 1 else 0.0
        skew = max_degree / max(avg_degree, 1e-8)
        
        num_components = len({find(x) for x in deg.keys()})
        density = (
            (2.0 * n_edges) / (n_vertices * (n_vertices - 1))
            if n_vertices > 1
            else 0.0
        )
        
        sample_graph = nx.Graph()
        sample_graph.add_edges_from(reservoir)
        
        clustering = (
            float(nx.average_clustering(sample_graph))
            if sample_graph.number_of_edges()
            else 0.0
        )
        diameter = int(self.extractor._compute_diameter(sample_graph))
        
        return {
            'graph_features': {
                'basic': {
                    'num_vertices': int(n_vertices),
                    'num_edges': int(n_edges),
                    'density': float(density),
                },
                'degree': {
                    'avg': float(avg_degree),
                    'max': float(max_degree),
                    'min': float(min_degree),
                    'std': float(degree_std),
                    'skew': float(skew),
                },
                'structure': {
                    'diameter': diameter,
                    'clustering_coeff': clustering,
                    'num_components': int(num_components),
                },
                'partition': {
                    'num_partitions': 1,
                    'boundary_vertices': 0,
                    'edge_cut_ratio': 0.0,
                    'balance': 1.0,
                },
            },
            'metadata': {
                'input_file': str(edge_file),
                'input_type': 'single_graph',
                'num_edges_loaded': int(n_edges),
                'extraction': 'streaming_single_pass',
                'structure_reservoir_edges': int(len(reservoir)),
            },
        }
    
    def extract_single(
        self,
        edge_file: str
    ) -> Dict[str, Any]:
        """
        Extract features from a single graph.
        
        Args:
            edge_file: Path to edge list file
            
        Returns:
            Dictionary of features
        """
        fp = Path(edge_file)
        if not fp.is_file():
            raise FileNotFoundError(edge_file)
        if fp.stat().st_size >= self._STREAMING_SIZE_THRESHOLD:
            return self._extract_single_streaming(edge_file)
        
        # Load edges
        edges = self.load_edge_list(edge_file)
        
        # Build graph
        graph = self.build_graph(edges)
        
        # Extract features
        features = self.extractor.extract(graph)
        feature_names = self.extractor.get_feature_names()
        
        # Build result
        result = {
            'graph_features': {
                'basic': {
                    'num_vertices': int(features[0]),
                    'num_edges': int(features[1]),
                    'density': float(nx.density(graph)) if nx else 0.0,
                },
                'degree': {
                    'avg': float(features[3]),
                    'max': float(features[4]),
                    'min': float(features[5]),
                    'std': float(features[6]),
                    'skew': float(features[7]),
                },
                'structure': {
                    'diameter': int(features[2]),
                    'clustering_coeff': float(features[8]),
                    'num_components': int(features[9]),
                },
                'partition': {
                    'num_partitions': 1,
                    'boundary_vertices': 0,
                    'edge_cut_ratio': 0.0,
                    'balance': 1.0,
                }
            },
            'metadata': {
                'input_file': str(edge_file),
                'input_type': 'single_graph',
                'num_edges_loaded': len(edges),
            }
        }
        
        return result
    
    def extract_partitioned(
        self,
        partition_folder: str
    ) -> Dict[str, Any]:
        """
        Extract features from partitioned graph.
        
        Args:
            partition_folder: Path to folder containing partition edge files
            
        Returns:
            Dictionary of features
        """
        folder = Path(partition_folder)
        
        # Find all edge files
        edge_files = []
        for ext in ['*.csv', '*.edges', '*.txt']:
            edge_files.extend(folder.glob(ext))
        
        if not edge_files:
            raise ValueError(f"No edge files found in {partition_folder}")
        
        # Sort files for consistent ordering
        edge_files = sorted(edge_files)
        
        # Load all partitions
        partitions = {}
        all_edges = []
        vertex_to_partition = {}
        
        for i, edge_file in enumerate(edge_files):
            edges = self.load_edge_list(str(edge_file))
            all_edges.extend(edges)
            
            # Track which partition each vertex belongs to
            for src, dst in edges:
                vertex_to_partition[src] = i
                vertex_to_partition[dst] = i
            
            partitions[i] = edges
        
        # Build global graph
        global_graph = self.build_graph(all_edges)
        
        # Build vertex sets for each partition
        partition_vertices = {}
        for partition_id, edges in partitions.items():
            vertices = set()
            for src, dst in edges:
                vertices.add(src)
                vertices.add(dst)
            partition_vertices[partition_id] = list(vertices)
        
        # Extract features with partitions
        features = self.extractor.extract(global_graph, partition_vertices)
        feature_names = self.extractor.get_feature_names()
        
        # Compute partition quality metrics
        edge_cut = PartitionQualityMetrics.edge_cut_ratio(global_graph, partition_vertices)
        balance = PartitionQualityMetrics.balance_score(partition_vertices)
        boundary = PartitionQualityMetrics.boundary_ratio(global_graph, partition_vertices)
        
        # Build result
        result = {
            'graph_features': {
                'basic': {
                    'num_vertices': int(features[0]),
                    'num_edges': int(features[1]),
                    'density': float(nx.density(global_graph)) if nx else 0.0,
                },
                'degree': {
                    'avg': float(features[3]),
                    'max': float(features[4]),
                    'min': float(features[5]),
                    'std': float(features[6]),
                    'skew': float(features[7]),
                },
                'structure': {
                    'diameter': int(features[2]),
                    'clustering_coeff': float(features[8]),
                    'num_components': int(features[9]),
                },
                'partition': {
                    'num_partitions': int(features[10]),
                    'boundary_vertices': int(features[11]),
                    'boundary_degree_sum': float(features[12]),
                    'avg_partition_size': float(features[13]),
                    'partition_size_std': float(features[14]),
                    'edge_cut_ratio': float(features[15]),
                    'balance': float(features[16]),
                },
                'quality_metrics': {
                    'edge_cut_ratio': float(edge_cut),
                    'balance_score': float(balance),
                    'boundary_ratio': float(boundary),
                    'comprehensive_score': float(
                        PartitionQualityMetrics.comprehensive_score(
                            global_graph, partition_vertices
                        )
                    ),
                }
            },
            'partition_info': {
                str(i): {
                    'file': str(f),
                    'num_edges': len(edges),
                    'num_vertices': len(set(v for e in edges for v in e)),
                }
                for i, (f, edges) in enumerate(zip(edge_files, [self.load_edge_list(str(f)) for f in edge_files]))
            },
            'metadata': {
                'input_folder': str(partition_folder),
                'input_type': 'partitioned_graph',
                'num_partitions': len(edge_files),
                'total_edges': len(all_edges),
            }
        }
        
        return result
    
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
        description='Extract features from graph data (edge list format)'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input edge list file or folder (for partitioned graphs)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='out/graph_features.yaml',
        help='Output YAML file path'
    )
    
    args = parser.parse_args()
    
    # Determine input type
    input_path = Path(args.input)
    
    extractor = GraphFeatureExtractor()
    
    if input_path.is_file():
        # Single graph
        print(f"Processing single graph: {args.input}")
        features = extractor.extract_single(args.input)
    elif input_path.is_dir():
        # Partitioned graph
        print(f"Processing partitioned graph: {args.input}")
        features = extractor.extract_partitioned(args.input)
    else:
        print(f"Error: Input path does not exist: {args.input}")
        return
    
    # Save to YAML
    extractor.save_to_yaml(features, args.output)
    
    # Print summary
    gf = features['graph_features']
    print(f"\nGraph features extracted:")
    print(f"  Vertices: {gf['basic']['num_vertices']}")
    print(f"  Edges: {gf['basic']['num_edges']}")
    print(f"  Density: {gf['basic']['density']:.4f}")
    print(f"  Avg degree: {gf['degree']['avg']:.2f}")
    print(f"  Diameter: {gf['structure']['diameter']}")
    print(f"  Partitions: {gf['partition']['num_partitions']}")
    print(f"  Balance: {gf['partition']['balance']:.4f}")
    print(f"  Edge cut ratio: {gf['partition']['edge_cut_ratio']:.4f}")
    print(f"  Output: {args.output}")


if __name__ == '__main__':
    main()
