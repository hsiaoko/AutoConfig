"""
Main pipeline for training and prediction.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple
import argparse

from autoconfig import CostPredictor, BayesianExecutionTimeModel


def generate_synthetic_data(
    n_samples: int = 100,
    n_query_nodes_range: Tuple[int, int] = (3, 10),
    n_graph_nodes_range: Tuple[int, int] = (50, 200),
    edge_probability: float = 0.1
) -> Tuple[List, List, List, np.ndarray]:
    """
    Generate synthetic training data for demonstration.
    
    Args:
        n_samples: Number of samples to generate
        n_query_nodes_range: Range of query graph sizes
        n_graph_nodes_range: Range of data graph sizes
        edge_probability: Probability of edge creation
        
    Returns:
        Tuple of (queries, graphs, configs, execution_times)
    """
    queries = []
    graphs = []
    configs = []
    execution_times = []
    
    np.random.seed(42)
    
    for i in range(n_samples):
        # Generate random query graph
        n_query_nodes = np.random.randint(*n_query_nodes_range)
        query = nx.erdos_renyi_graph(n_query_nodes, edge_probability * 2)
        
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
        
        # Generate synthetic execution time
        # (in real scenario, this would be actual measured time)
        base_time = (
            n_query_nodes * 10 +
            n_graph_nodes * 0.5 +
            query.number_of_edges() * 5 +
            graph.number_of_edges() * 0.1 +
            (16 / config['num_threads']) * 100 +
            (16384 / config['memory_limit']) * 50
        )
        noise = np.random.normal(0, base_time * 0.1)
        exec_time = max(1.0, base_time + noise)
        
        queries.append(query)
        graphs.append(graph)
        configs.append(config)
        execution_times.append(exec_time)
    
    return queries, graphs, configs, np.array(execution_times)


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
        # For demonstration, use untrained model (predictions will be poor)
        print("Warning: Using untrained model for demonstration")
    
    # Create a sample query
    query = nx.Graph()
    query.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3)])  # 4-node cycle
    
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
        query, graph, config, return_uncertainty=True
    )
    
    print("\nPrediction Example:")
    print(f"  Query: {query.number_of_nodes()} nodes, {query.number_of_edges()} edges")
    print(f"  Data Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"  Predicted Execution Time: {pred:.2f} ms")
    print(f"  95% Confidence Interval: [{lower:.2f}, {upper:.2f}] ms")


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
        # Default: run both training and prediction
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


if __name__ == '__main__':
    main()
