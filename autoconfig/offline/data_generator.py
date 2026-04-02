"""
Data Generator for Training and Prediction

Generates synthetic data for:
- Query configurations
- Graph structures
- Execution time/cost labels
"""

import numpy as np
import networkx as nx
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime


class DataGenerator:
    """
    Generate synthetic training/prediction data.
    
    Generates:
    - Query configurations (different algorithm patterns)
    - Graph structures (various sizes and properties)
    - System configurations (resource allocations)
    - Execution time and cost labels
    """
    
    # Query templates with different computational patterns
    QUERY_TEMPLATES = {
        'bfs': {
            'code': '''
def BFS(Graph G, vertex source):
    worklist = [source]
    visited[source] = true
    while !worklist.empty():
        for v in worklist:
            for neighbor in G.neighbors(v):
                if !visited[neighbor]:
                    visited[neighbor] = true
                    worklist.append(neighbor)
''',
            'complexity': {'v_scan': 1, 'e_scan': 1, 'f_scan': 1, 'atomic': 0, 'sync': 0},
            'base_cost': 50.0,
        },
        'dfs': {
            'code': '''
def DFS(Graph G, vertex source):
    stack = [source]
    visited[source] = true
    while !stack.empty():
        v = stack.pop()
        for neighbor in G.neighbors(v):
            if !visited[neighbor]:
                visited[neighbor] = true
                stack.append(neighbor)
''',
            'complexity': {'v_scan': 1, 'e_scan': 1, 'f_scan': 0, 'atomic': 0, 'sync': 0},
            'base_cost': 45.0,
        },
        'pagerank': {
            'code': '''
def PageRank(Graph G, int iterations):
    for i in 1..iterations:
        for v in G.vertices():
            rank[v] = 0
        for v in G.vertices():
            for neighbor in G.neighbors(v):
                rank[neighbor] += rank[v] / out_degree[v]
        barrier()
''',
            'complexity': {'v_scan': 2, 'e_scan': 1, 'f_scan': 0, 'atomic': 0, 'sync': 1},
            'base_cost': 150.0,
        },
        'cc': {
            'code': '''
def ConnectedComponents(Graph G):
    component = [0..|V|]
    changed = true
    while changed:
        changed = false
        for v in G.vertices():
            for neighbor in G.neighbors(v):
                if component[v] < component[neighbor]:
                    component[neighbor] = component[v]
                    changed = true
        barrier()
''',
            'complexity': {'v_scan': 1, 'e_scan': 1, 'f_scan': 0, 'atomic': 0, 'sync': 1},
            'base_cost': 120.0,
        },
        'sssp': {
            'code': '''
def SSSP(Graph G, vertex source, weight W):
    dist = [infinity]
    dist[source] = 0
    worklist = [source]
    while !worklist.empty():
        for v in worklist:
            for neighbor in G.neighbors(v):
                new_dist = dist[v] + W[v, neighbor]
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    worklist.append(neighbor)
''',
            'complexity': {'v_scan': 1, 'e_scan': 1, 'f_scan': 1, 'atomic': 1, 'sync': 0},
            'base_cost': 100.0,
        },
        'kcore': {
            'code': '''
def KCore(Graph G, int k):
    core = [true]
    changed = true
    while changed:
        changed = false
        for v in G.vertices():
            if core[v] and degree[v] < k:
                core[v] = false
                changed = true
        barrier()
''',
            'complexity': {'v_scan': 1, 'e_scan': 0, 'f_scan': 0, 'atomic': 0, 'sync': 1},
            'base_cost': 80.0,
        },
        'tc': {
            'code': '''
def TriangleCounting(Graph G):
    count = 0
    for v in G.vertices():
        for neighbor1 in G.neighbors(v):
            if neighbor1 > v:
                for neighbor2 in G.neighbors(v):
                    if neighbor2 > neighbor1 and G.has_edge(neighbor1, neighbor2):
                        atomicAdd(count, 1)
''',
            'complexity': {'v_scan': 1, 'e_scan': 2, 'f_scan': 0, 'atomic': 1, 'sync': 0},
            'base_cost': 200.0,
        },
    }
    
    # Resource configuration templates
    RESOURCE_CONFIGS = [
        {'cpu_cores': 4, 'memory_gb': 16, 'num_gpus': 0, 'gpu_memory_gb': 0, 'storage_gb': 100},
        {'cpu_cores': 8, 'memory_gb': 32, 'num_gpus': 0, 'gpu_memory_gb': 0, 'storage_gb': 200},
        {'cpu_cores': 16, 'memory_gb': 64, 'num_gpus': 1, 'gpu_memory_gb': 16, 'storage_gb': 500},
        {'cpu_cores': 32, 'memory_gb': 128, 'num_gpus': 2, 'gpu_memory_gb': 32, 'storage_gb': 1000},
        {'cpu_cores': 64, 'memory_gb': 256, 'num_gpus': 4, 'gpu_memory_gb': 80, 'storage_gb': 2000},
        {'cpu_cores': 128, 'memory_gb': 512, 'num_gpus': 8, 'gpu_memory_gb': 80, 'storage_gb': 5000},
    ]
    
    def __init__(self, seed: int = 42):
        """Initialize data generator with random seed."""
        self.seed = seed
        np.random.seed(seed)
        self.query_templates = self.QUERY_TEMPLATES
        self.resource_configs = self.RESOURCE_CONFIGS
        
    def generate_query(self, query_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a query configuration.
        
        Args:
            query_name: Specific query name or None for random
            
        Returns:
            Query dictionary with code and metadata
        """
        if query_name is None:
            query_name = np.random.choice(list(self.query_templates.keys()))
        
        template = self.query_templates[query_name]
        return {
            'query_name': query_name,
            'code': template['code'].strip(),
            'complexity': template['complexity'].copy(),
            'base_cost': template['base_cost'],
        }
    
    def generate_graph(
        self,
        num_vertices: int = 1000,
        edge_probability: float = 0.01,
        graph_type: str = 'erdos_renyi',
        num_partitions: int = 1
    ) -> Tuple[nx.Graph, Dict[str, Any]]:
        """
        Generate a graph with specified properties.
        
        Args:
            num_vertices: Number of vertices
            edge_probability: Probability of edge creation
            graph_type: Type of graph ('erdos_renyi', 'powerlaw', 'community')
            num_partitions: Number of partitions (for partitioned graphs)
            
        Returns:
            Tuple of (NetworkX graph, graph features dict)
        """
        if graph_type == 'erdos_renyi':
            graph = nx.erdos_renyi_graph(num_vertices, edge_probability)
        elif graph_type == 'powerlaw':
            graph = nx.powerlaw_cluster_graph(num_vertices, int(edge_probability * num_vertices), 0.5)
        elif graph_type == 'community':
            # Generate community structure
            num_communities = max(1, num_vertices // 50)
            graph = nx.planted_partition_graph(num_communities, num_vertices // num_communities, 
                                               edge_probability * 2, edge_probability * 0.2)
        else:
            graph = nx.erdos_renyi_graph(num_vertices, edge_probability)
        
        # Compute graph features
        degrees = [d for _, d in graph.degree()]
        features = {
            'num_vertices': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'avg_degree': np.mean(degrees) if degrees else 0,
            'max_degree': max(degrees) if degrees else 0,
            'min_degree': min(degrees) if degrees else 0,
            'std_degree': np.std(degrees) if degrees else 0,
            'density': nx.density(graph),
            'avg_clustering': nx.average_clustering(graph),
            'num_partitions': num_partitions,
            'graph_type': graph_type,
        }
        
        return graph, features
    
    def generate_config(
        self,
        config_idx: Optional[int] = None,
        perturb: bool = False,
        perturb_range: int = 1
    ) -> Dict[str, Any]:
        """
        Generate a system configuration.
        
        Args:
            config_idx: Specific config index or None for random
            perturb: Whether to perturb a base config
            perturb_range: Range of perturbation for core count
            
        Returns:
            Configuration dictionary
        """
        if config_idx is None:
            config_idx = np.random.randint(len(self.resource_configs))
        
        base_config = self.resource_configs[config_idx].copy()
        
        if perturb:
            # Perturb CPU cores
            delta = np.random.randint(-perturb_range, perturb_range + 1)
            base_config['cpu_cores'] = max(1, base_config['cpu_cores'] + delta * 4)
        
        config_id = f"config_{config_idx}"
        if perturb:
            config_id += f"_p{delta}"
        
        return {
            'config_id': config_id,
            'resource': base_config,
            'k': 1,  # Number of instances (can be scaled)
        }
    
    def compute_execution_time(
        self,
        query: Dict[str, Any],
        graph_features: Dict[str, Any],
        config: Dict[str, Any],
        noise_factor: float = 0.1
    ) -> float:
        """
        Compute synthetic execution time based on query, graph, and config.
        
        Uses a cost model that considers:
        - Query complexity (vertex/edge scans, atomics, sync)
        - Graph size and structure
        - Resource capacity
        
        Args:
            query: Query dictionary
            graph_features: Graph features dictionary
            config: Configuration dictionary
            noise_factor: Amount of noise to add
            
        Returns:
            Estimated execution time in milliseconds
        """
        resource = config['resource']
        complexity = query['complexity']
        
        # Base time from query
        base_time = query['base_cost']
        
        # Graph size factor (log scale for vertices, linear for edges)
        V = graph_features['num_vertices']
        E = graph_features['num_edges']
        graph_factor = (
            complexity['v_scan'] * np.log(V + 1) * 10 +
            complexity['e_scan'] * E / V * 5 +
            complexity['f_scan'] * V * 0.05
        )
        
        # Resource factor (inverse relationship - more resources = faster)
        cpu_factor = 32 / resource['cpu_cores']
        mem_factor = 64 / resource['memory_gb']
        gpu_factor = 1.0
        if resource['num_gpus'] > 0:
            # GPU acceleration for certain operations
            gpu_factor = 0.5 / resource['num_gpus']
        
        resource_factor = (cpu_factor + mem_factor) / 2 * gpu_factor
        
        # Synchronization overhead
        sync_overhead = complexity['sync'] * 20
        
        # Atomic operation overhead
        atomic_overhead = complexity['atomic'] * E * 0.001
        
        # Compute final time
        time = (base_time + graph_factor) * resource_factor + sync_overhead + atomic_overhead
        
        # Add noise
        noise = np.random.normal(0, time * noise_factor)
        time = max(1.0, time + noise)
        
        return time
    
    def compute_execution_cost(
        self,
        execution_time: float,
        config: Dict[str, Any]
    ) -> float:
        """
        Compute execution cost (monetary/resource cost).
        
        Cost model based on:
        - Execution duration
        - Resource type and count
        - Cloud pricing approximation
        
        Args:
            execution_time: Execution time in milliseconds
            config: Configuration dictionary
            
        Returns:
            Execution cost (arbitrary units, proportional to $)
        """
        resource = config['resource']
        k = config.get('k', 1)
        
        # Hourly rates (approximate cloud pricing)
        cpu_hourly = 0.05  # per core
        memory_hourly = 0.01  # per GB
        gpu_hourly = 0.50  # per GPU
        
        # Convert ms to hours
        duration_hours = execution_time / 1000 / 3600
        
        # Compute cost
        cpu_cost = resource['cpu_cores'] * cpu_hourly * duration_hours
        mem_cost = resource['memory_gb'] * memory_hourly * duration_hours
        gpu_cost = resource['num_gpus'] * gpu_hourly * duration_hours
        
        total_cost = (cpu_cost + mem_cost + gpu_cost) * k * 1000  # Scale for readability
        
        return total_cost
    
    def generate_sample(
        self,
        query_name: Optional[str] = None,
        graph_params: Optional[Dict[str, Any]] = None,
        config_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete sample (query + graph + config + labels).
        
        Args:
            query_name: Query type name
            graph_params: Graph generation parameters
            config_idx: Configuration index
            
        Returns:
            Sample dictionary with all features and labels
        """
        # Generate components
        query = self.generate_query(query_name)
        
        graph_params = graph_params or {}
        graph, graph_features = self.generate_graph(**graph_params)
        
        config = self.generate_config(config_idx)
        
        # Compute labels
        exec_time = self.compute_execution_time(query, graph_features, config)
        exec_cost = self.compute_execution_cost(exec_time, config)
        
        return {
            'query': query,
            'graph': graph,
            'graph_features': graph_features,
            'config': config,
            'labels': {
                'execution_time_ms': exec_time,
                'execution_cost': exec_cost,
            }
        }
    
    def generate_dataset(
        self,
        num_samples: int = 100,
        query_names: Optional[List[str]] = None,
        graph_size_range: Tuple[int, int] = (100, 5000),
        edge_prob_range: Tuple[float, float] = (0.001, 0.05),
        config_indices: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate a dataset of samples.
        
        Args:
            num_samples: Number of samples to generate
            query_names: List of query names to use
            graph_size_range: Range of graph sizes
            edge_prob_range: Range of edge probabilities
            config_indices: List of config indices to use
            
        Returns:
            List of sample dictionaries
        """
        if query_names is None:
            query_names = list(self.query_templates.keys())
        
        if config_indices is None:
            config_indices = list(range(len(self.resource_configs)))
        
        dataset = []
        
        for i in range(num_samples):
            # Random selections
            query_name = np.random.choice(query_names)
            num_vertices = np.random.randint(*graph_size_range)
            edge_prob = np.random.uniform(*edge_prob_range)
            config_idx = np.random.choice(config_indices)
            
            graph_params = {
                'num_vertices': num_vertices,
                'edge_probability': edge_prob,
                'graph_type': np.random.choice(['erdos_renyi', 'powerlaw', 'community']),
            }
            
            sample = self.generate_sample(query_name, graph_params, config_idx)
            dataset.append(sample)
        
        return dataset
    
    def save_dataset(
        self,
        dataset: List[Dict[str, Any]],
        output_path: str,
        save_graphs: bool = False
    ):
        """
        Save dataset to files.
        
        Args:
            dataset: List of samples
            output_path: Output directory or file path
            save_graphs: Whether to save graph structures
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata and features
        records = []
        for i, sample in enumerate(dataset):
            record = {
                'sample_id': i,
                'query_name': sample['query']['query_name'],
                'query_complexity': sample['query']['complexity'],
                'graph_features': sample['graph_features'],
                'config': sample['config'],
                'labels': sample['labels'],
            }
            records.append(record)
        
        # Save as YAML
        output_file = output_dir / 'dataset.yaml'
        with open(output_file, 'w') as f:
            yaml.dump({
                'metadata': {
                    'num_samples': len(dataset),
                    'generated_at': datetime.now().isoformat(),
                    'seed': self.seed,
                },
                'samples': records,
            }, f, default_flow_style=None)
        
        # Optionally save graphs as edge lists
        if save_graphs:
            graphs_dir = output_dir / 'graphs'
            graphs_dir.mkdir(exist_ok=True)
            
            for i, sample in enumerate(dataset):
                graph = sample['graph']
                edge_file = graphs_dir / f'graph_{i}.csv'
                with open(edge_file, 'w') as f:
                    f.write('src,dst\n')
                    for src, dst in graph.edges():
                        f.write(f'{src},{dst}\n')
        
        print(f"Dataset saved to {output_dir}")
        print(f"  Samples: {len(dataset)}")
        print(f"  Output: {output_file}")
        if save_graphs:
            print(f"  Graphs: {graphs_dir}")
    
    @classmethod
    def load_dataset(cls, input_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Load dataset from file.
        
        Args:
            input_path: Path to dataset YAML file
            
        Returns:
            Tuple of (samples list, metadata dict)
        """
        with open(input_path, 'r') as f:
            data = yaml.safe_load(f)
        
        metadata = data.get('metadata', {})
        samples = data.get('samples', [])
        
        return samples, metadata
