"""
Main pipeline for training and prediction.

Supports both code-based and graph-based query representation.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple
import argparse

from autoconfig import CostPredictor, BayesianExecutionTimeModel


# Sample query code templates
QUERY_TEMPLATES = {
    'vertex_scan': """
for v in G.vertices():
    process(v)
""",
    'edge_scan': """
for v in G.vertices():
    for neighbor in G.neighbors(v):
        process(v, neighbor)
""",
    'frontier': """
worklist = [source]
while !worklist.empty():
    for v in worklist:
        for neighbor in G.neighbors(v):
            visit(neighbor)
""",
    'recursive': """
def expand(node, depth):
    if depth == 0: return
    for neighbor in G.neighbors(node):
        expand(neighbor, depth - 1)
""",
    'atomic': """
for edge in G.edges():
    atomicAdd(counter[edge.src], 1)
""",
    'complex': """
for v in G.vertices():
    if condition(v):
        for neighbor in G.neighbors(v):
            atomicAdd(rank[neighbor], 1)
    barrier()
""",
}


def generate_synthetic_data(
    n_samples: int = 100,
    n_graph_nodes_range: Tuple[int, int] = (50, 200),
    edge_probability: float = 0.1
) -> Tuple[List[str], List[nx.Graph], List[Dict], np.ndarray]:
    """
    Generate synthetic training data for demonstration.
    
    Uses code-based query representation.
    
    Args:
        n_samples: Number of samples to generate
        n_graph_nodes_range: Range of data graph sizes
        edge_probability: Probability of edge creation
        
    Returns:
        Tuple of (query_codes, graphs, configs, execution_times)
    """
    query_codes = []
    graphs = []
    configs = []
    execution_times = []
    
    np.random.seed(42)
    
    query_template_names = list(QUERY_TEMPLATES.keys())
    
    for i in range(n_samples):
        # Select random query template
        template_name = np.random.choice(query_template_names)
        query_code = QUERY_TEMPLATES[template_name]
        
        # Generate random data graph
        n_graph_nodes = np.random.randint(*n_graph_nodes_range)
        graph = nx.erdos_renyi_graph(n_graph_nodes, edge_probability)
        
        # Generate random configuration
        config = {
            'memory_limit': np.random.choice([4096, 8192, 16384]),
            'num_threads': np.random.choice([2, 4, 8, 16]),
            'cache_size': np.random.choice([512, 1024, 2048]),
            'batch_size': np.random.choice([500, 1000, 2000]),
            'io_buffer_size': np.random.choice([32, 64, 128]),
            'num_workers': np.random.choice([1, 2, 4]),
            'timeout': 300,
            'enable_index': np.random.choice([True, False]),
            'index_type': np.random.choice(['btree', 'hash', 'bitmap']),
            'compression_enabled': np.random.choice([True, False])
        }
        
        # Generate synthetic execution time based on features
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        degrees = [d for _, d in graph.degree()]
        max_degree = max(degrees) if degrees else 0
        
        # Base time from graph size
        base_time = n_nodes * 0.5 + n_edges * 0.1
        
        # Add query complexity
        complexity_bonus = {
            'vertex_scan': 10,
            'edge_scan': 50,
            'frontier': 100,
            'recursive': 200,
            'atomic': 80,
            'complex': 150,
        }
        base_time += complexity_bonus.get(template_name, 50)
        
        # Configuration effects
        config_factor = (
            (16 / config['num_threads']) * 20 +
            (16384 / config['memory_limit']) * 10
        )
        base_time += config_factor
        
        # Add noise
        noise = np.random.normal(0, base_time * 0.1)
        exec_time = max(1.0, base_time + noise)
        
        query_codes.append(query_code)
        graphs.append(graph)
        configs.append(config)
        execution_times.append(exec_time)
    
    return query_codes, graphs, configs, np.array(execution_times)


def train_pipeline(
    n_train_samples: int = 80,
    n_test_samples: int = 20,
    verbose: bool = True
) -> CostPredictor:
    """
    Run training pipeline.
    
    Args:
        n_train_samples: Number of training samples
        n_test_samples: Number of test samples
        verbose: Whether to print progress
        
    Returns:
        Trained CostPredictor
    """
    if verbose:
        print("Generating synthetic data...")
    
    # Generate data
    queries, graphs, configs, exec_times = generate_synthetic_data(
        n_train_samples + n_test_samples
    )
    
    # Split data
    train_queries = queries[:n_train_samples]
    train_graphs = graphs[:n_train_samples]
    train_configs = configs[:n_train_samples]
    train_times = exec_times[:n_train_samples]
    
    test_queries = queries[n_train_samples:]
    test_graphs = graphs[n_train_samples:]
    test_configs = configs[n_train_samples:]
    test_times = exec_times[n_train_samples:]
    
    # Initialize predictor
    predictor = CostPredictor()
    
    # Train
    if verbose:
        print("\nTraining model...")
    
    train_metrics = predictor.train(
        train_queries, train_graphs, train_configs, train_times,
        verbose=verbose
    )
    
    # Evaluate
    if verbose:
        print("\nEvaluating model...")
    
    test_metrics = predictor.evaluate(
        test_queries, test_graphs, test_configs, test_times
    )
    
    if verbose:
        print("\nTest Results:")
        print(f"  MAE: {test_metrics['mae']:.4f}")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  MAPE: {test_metrics['mape']:.2f}%")
        print(f"  R²: {test_metrics['r2']:.4f}")
        print(f"  Calibration (95% CI): {test_metrics['calibration']:.2%}")
    
    return predictor


def prediction_example(predictor: CostPredictor = None):
    """
    Demonstrate prediction with a new query-graph-config combination.
    
    Args:
        predictor: Trained CostPredictor (optional, will create one if None)
    """
    # Create or load predictor
    if predictor is None:
        predictor = CostPredictor()
        print("Warning: Using untrained model for demonstration")
    
    # Sample query code (BFS-like)
    query_code = """
    BFS(Graph G, source):
        worklist = [source]
        visited[source] = true
        while !worklist.empty():
            for v in worklist:
                for neighbor in G.neighbors(v):
                    if !visited[neighbor]:
                        visited[neighbor] = true
                        worklist.append(neighbor)
    """
    
    # Create a sample data graph
    graph = nx.erdos_renyi_graph(100, 0.1)
    
    # Define configuration
    config = {
        'memory_limit': 8192,
        'num_threads': 4,
        'cache_size': 1024,
        'batch_size': 1000,
        'io_buffer_size': 64,
        'num_workers': 2,
        'timeout': 300,
        'enable_index': True,
        'index_type': 'btree',
        'compression_enabled': False
    }
    
    # Predict
    pred, lower, upper = predictor.predict(
        query_code, graph, config, return_uncertainty=True
    )
    
    print("\nPrediction Example:")
    print(f"  Query: BFS (code-based)")
    print(f"  Data Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"  Predicted Execution Time: {pred:.2f} ms")
    print(f"  95% Confidence Interval: [{lower:.2f}, {upper:.2f}] ms")


def feature_demo():
    """Demonstrate feature extraction."""
    from autoconfig import FeatureManager
    
    print("\n" + "=" * 60)
    print("Feature Extraction Demo")
    print("=" * 60)
    
    manager = FeatureManager()
    
    # Sample query
    query_code = QUERY_TEMPLATES['edge_scan']
    graph = nx.erdos_renyi_graph(50, 0.1)
    config = {'memory_limit': 8192, 'num_threads': 4}
    
    features = manager.extract_all(query_code, graph, config)
    dims = manager.get_feature_dimensions()
    groups = manager.get_feature_groups()
    
    print(f"\nTotal features: {len(features)}")
    print(f"Feature dimensions: {dims}")
    
    print("\nFeature groups:")
    for group_name, names in groups.items():
        print(f"  {group_name}: {len(names)} features")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Graph Query Execution Time Prediction'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Run training pipeline'
    )
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Run prediction example'
    )
    parser.add_argument(
        '--features',
        action='store_true',
        help='Run feature extraction demo'
    )
    parser.add_argument(
        '--n-train',
        type=int,
        default=80,
        help='Number of training samples'
    )
    parser.add_argument(
        '--n-test',
        type=int,
        default=20,
        help='Number of test samples'
    )
    parser.add_argument(
        '--save-model',
        type=str,
        default=None,
        help='Path to save trained model'
    )
    
    args = parser.parse_args()
    
    if args.features:
        feature_demo()
        return
    
    if args.train:
        predictor = train_pipeline(
            n_train_samples=args.n_train,
            n_test_samples=args.n_test
        )
        
        if args.save_model:
            predictor.save_model(args.save_model)
            print(f"\nModel saved to {args.save_model}")
        
        # Run prediction example with trained model
        prediction_example(predictor)
    
    elif args.predict:
        prediction_example()
    
    else:
        # Default: run training, evaluation, and prediction
        print("=" * 60)
        print("Graph Query Execution Time Prediction System")
        print("=" * 60)
        
        predictor = train_pipeline(
            n_train_samples=args.n_train,
            n_test_samples=args.n_test
        )
        
        if args.save_model:
            predictor.save_model(args.save_model)
            print(f"\nModel saved to {args.save_model}")
        
        prediction_example(predictor)
        feature_demo()


if __name__ == '__main__':
    main()
