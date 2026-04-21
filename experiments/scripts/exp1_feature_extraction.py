#!/usr/bin/env python3
"""
Exp-1: Feature Extraction Effectiveness
Evaluates whether the unified feature abstraction improves cost estimation quality.

Components:
(1) In-distribution (seen tasks)
(2) Out-of-distribution (unseen tasks) - transfer learning
"""

import os
import sys
import numpy as np
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from autoconfig import CostPredictor, FeatureManager
from autoconfig.models import BayesianExecutionTimeModel
from autoconfig.feature_extractor import static_extractor, graph_extractor, symbolic_extractor

# Add workloads to path and import
sys.path.insert(0, str(Path(__file__).parent.parent / "workloads"))
from workloads import query_wcc, query_sssp, query_pr, query_bfs, query_subiso


def generate_training_data(
    n_samples: int,
    workload_name: str,
    graph_size_range: Tuple[int, int] = (1000, 10000),
    config_variety: int = 50
):
    """
    Generate synthetic training data for a specific workload.

    Args:
        n_samples: Number of samples to generate
        workload_name: Name of the workload (WCC, SSSP, PR, BFS, SubIso)
        graph_size_range: Range of graph sizes (min, max nodes)
        config_variety: Number of distinct configurations to sample

    Returns:
        (queries, graphs, configs, execution_times)
    """
    import networkx as nx

    workload_queries = {
        'WCC': query_wcc,
        'SSSP': query_sssp,
        'PR': query_pr,
        'BFS': query_bfs,
        'SubIso': query_subiso,
    }

    query_code = workload_queries[workload_name]

    queries = []
    graphs = []
    configs = []
    execution_times = []

    np.random.seed(42)

    # LHS sampling for configurations
    k_values = np.linspace(1, 64, config_variety).astype(int)

    for i in range(n_samples):
        queries.append(query_code)

        # Generate graph with varying characteristics
        n_graph_nodes = np.random.randint(*graph_size_range)

        # Vary graph structure params
        edge_prob = np.random.uniform(0.05, 0.2)
        graph_generators = {
            'erdos_renyi': lambda: nx.erdos_renyi_graph(n_graph_nodes, edge_prob),
            'barabasi_albert': lambda: nx.barabasi_albert_graph(n_graph_nodes, 2),
            'watts_strogatz': lambda: nx.watts_strogatz_graph(n_graph_nodes, 6, 0.1),
        }
        gen_name = np.random.choice(list(graph_generators.keys()))
        graph = graph_generators[gen_name]()
        graphs.append(graph)

        # Generate configuration
        idx = i % len(k_values)
        config = {
            'k': int(k_values[idx]),
            'cpu_cores': np.random.choice([16, 32, 64, 128]),
            'memory_gb': np.random.choice([64, 128, 256, 512]),
            'num_gpus': np.random.choice([0, 1, 2, 4, 8])
        }
        configs.append(config)

        # Generate execution time
        exec_time = _compute_exec_time(workload_name, graph, config, noise_level=0.15)
        execution_times.append(exec_time)

    return queries, graphs, configs, np.array(execution_times)


def _compute_exec_time(
    workload: str,
    graph: any,
    config: Dict[str, Any],
    noise_level: float = 0.15
) -> float:
    """
    Compute synthetic execution time based on workload, graph, and config.
    Uses realistic scaling factors.
    """
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    # Workload complexity
    complexity_multipliers = {
        'WCC': 1.0,      # Iterative fixpoint
        'SSSP': 2.0,     # Frontier-driven
        'PR': 1.5,       # Iterative fixpoint
        'BFS': 2.5,      # Frontier-driven
        'SubIso': 8.0,   # Recursive pattern matching (expensive)
    }

    # Resource effects
    cpu_factor = 128.0 / (config['cpu_cores'] + 1e-8)
    gpu_factor = 1.0 if config['num_gpus'] == 0 else (8.0 / (config['num_gpus'] + 1e-8))
    k_factor = config['k'] / 2.0  # More instances -> faster but diminishing returns

    # Linear scaling with graph size
    base_time = (n_nodes * 0.01 + n_edges * 0.002) * complexity_multipliers[workload]

    # Combined resource effect
    resource_effect = cpu_factor * gpu_factor * k_factor

    # Final time with noise
    exec_time = base_time * resource_effect
    noise = np.random.normal(0, exec_time * noise_level)
    exec_time = max(1.0, exec_time + noise)

    return exec_time


def train_with_feature_variant(
    queries: List[str],
    graphs: List,
    configs: List[Dict],
    execution_times: np.ndarray,
    feature_variant: str,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Train model with different feature variants.

    Args:
        queries: List of query codes
        graphs: List of graphs
        configs: List of configurations
        execution_times: Execution times
        feature_variant: One of 'full', 'noSPF', 'noSGF', 'raw'
        verbose: Print progress

    Returns:
        Dictionary with metrics and model
    """
    predictor = CostPredictor()
    manager = predictor.feature_manager

    # Modify feature manager based on variant
    original_methods = {
        'static': manager.static_extractor,
        'symbolic': manager.symbolic_extractor,
        'graph': manager.graph_extractor
    }

    if feature_variant == 'noSPF':
        manager.static_extractor = None
    elif feature_variant == 'noSGF':
        manager.symbolic_extractor = None
    elif feature_variant == 'raw':
        manager.static_extractor = None
        manager.symbolic_extractor = None

    # Train model
    train_metrics = predictor.train(queries, graphs, configs, execution_times, verbose=verbose)

    # Restore
    manager.static_extractor = original_methods['static']
    manager.symbolic_extractor = original_methods['symbolic']
    manager.graph_extractor = original_methods['graph']

    return {
        'metrics': train_metrics,
        'model': predictor
    }


def evaluate_in_distribution(
    workload: str,
    output_dir: str,
    n_train: int = 600,
    n_test: int = 200
) -> Dict[str, Any]:
    """
    Evaluate in-distribution performance for a workload (60/20/20 split).

    Returns metrics for all feature variants.
    """
    print(f"\n{'='*60}")
    print(f"Exp-1 (1): In-distribution evaluation for {workload}")
    print(f"{'='*60}")

    # Generate data
    queries, graphs, configs, exec_times = generate_training_data(
        n_train + n_test, workload
    )

    # Split
    train_queries = queries[:n_train]
    train_graphs = graphs[:n_train]
    train_configs = configs[:n_train]
    train_times = exec_times[:n_train]

    test_queries = queries[n_train:]
    test_graphs = graphs[n_train:]
    test_configs = configs[n_train:]
    test_times = exec_times[n_train:]

    results = {}

    # Test each variant
    for variant in ['full', 'noSPF', 'noSGF', 'raw']:
        print(f"\nTraining variant: {variant}")

        result = train_with_feature_variant(
            train_queries, train_graphs, train_configs, train_times,
            variant, verbose=False
        )

        # Evaluate on test set
        test_metrics = result['model'].evaluate(
            test_queries, test_graphs, test_configs, test_times
        )

        results[variant] = {
            'train': result['metrics'],
            'test': test_metrics
        }

        print(f"  Test MAE: {test_metrics['mae']:.4f}")
        print(f"  Test MAPE: {test_metrics['mape']:.2f}%")
        print(f"  Test R²: {test_metrics['r2']:.4f}")

    # Save results
    output_path = Path(output_dir) / f"exp1_inv_dist_{workload}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


def evaluate_out_of_distribution(
    source_workloads: List[str],
    target_workload: str,
    transfer_name: str,
    output_dir: str,
    n_src_train: int = 800,
    n_tgt_test: int = 200
) -> Dict[str, Any]:
    """
    Evaluate transfer learning: train on source tasks, test on target task.
    """
    print(f"\n{'='*60}")
    print(f"Exp-1 (2): Out-of-distribution evaluation")
    print(f"  Transfer: {transfer_name}")
    print(f"  Source: {source_workloads} -> Target: {target_workload}")
    print(f"{'='*60}")

    # Generate source training data
    src_queries = []
    src_graphs = []
    src_configs = []
    src_times = []

    n_per_src = n_src_train // len(source_workloads)
    for i, src_wl in enumerate(source_workloads):
        q, g, c, t = generate_training_data(n_per_src, src_wl)
        src_queries.extend(q)
        src_graphs.extend(g)
        src_configs.extend(c)
        src_times.extend(t)

    src_questions = np.array(src_times)
    n_all = len(src_questions)

    # Split source data
    train_split = int(n_all * 0.8)
    train_src_queries = src_queries[:train_split]
    train_src_graphs = src_graphs[:train_split]
    train_src_configs = src_configs[:train_split]
    train_src_times = src_times[:train_split]

    val_src_queries = src_queries[train_split:]
    val_src_graphs = src_graphs[train_split:]
    val_src_configs = src_configs[train_split:]
    val_src_times = src_times[train_split:]

    # Generate target test data
    tgt_queries, tgt_graphs, tgt_configs, tgt_times = generate_training_data(
        n_tgt_test, target_workload
    )

    results = {}

    # Test each variant for transfer learning
    for variant in ['full', 'noSPF', 'noSGF', 'raw']:
        print(f"\nTraining variant: {variant}")

        result = train_with_feature_variant(
            train_src_queries, train_src_graphs, train_src_configs, train_src_times,
            variant, verbose=False
        )

        # Evaluate on target (OOD)
        ood_metrics = result['model'].evaluate(
            tgt_queries, tgt_graphs, tgt_configs, tgt_times
        )

        # Also evaluate on source validation set for comparison
        id_metrics = result['model'].evaluate(
            val_src_queries, val_src_graphs, val_src_configs, val_src_times
        )

        results[variant] = {
            'ood': ood_metrics,
            'id_self': id_metrics
        }

        print(f"  In-domain MAPE: {id_metrics['mape']:.2f}%")
        print(f"  Out-of-domain MAPE: {ood_metrics['mape']:.2f}%")
        print(f"  MAPE degradation: {ood_metrics['mape'] - id_metrics['mape']:.2f}%")

    # Save results
    output_path = Path(output_dir) / f"exp1_out_dist_{transfer_name.replace(' ', '_')}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results


def run_exp1(config: Dict[str, Any]):
    """
    Run complete Exp-1: Feature extraction effectiveness.
    """
    print(f"\n{'#'*70}")
    print(f"# Exp-1: Feature Extraction Effectiveness")
    print(f"{'#'*70}")

    output_dir = Path(config['global']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Part (1): In-distribution evaluation
    print("\n" + "="*70)
    print("Part (1): In-distribution (seen tasks) evaluation")
    print("="*70)

    workloads = ['WCC', 'SSSP', 'PR', 'BFS', 'SubIso']
    n_train = int(1000 * config['training']['train_ratio'])
    n_test = int(1000 * config['training']['test_ratio'])

    inv_dist_results = {}
    for wl in workloads:
        result = evaluate_in_distribution(wl, str(output_dir), n_train, n_test)
        inv_dist_results[wl] = result

    # Part (2): Out-of-distribution evaluation
    print("\n" + "="*70)
    print("Part (2): Out-of-distribution (unseen tasks) evaluation")
    print("="*70)

    ood_settings = config['experiments']['exp1']['transfer_settings']
    ood_results = {}

    for setting in ood_settings:
        result = evaluate_out_of_distribution(
            setting['source'],
            setting['target'],
            setting['name'],
            str(output_dir)
        )
        ood_results[setting['name']] = result

    # Summary
    print("\n" + "="*70)
    print("Exp-1 Summary")
    print("="*70)

    summary_path = output_dir / "exp1_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'in_distribution': inv_dist_results,
            'out_of_distribution': ood_results
        }, f, indent=2)

    print(f"\nAll results saved to {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == '__main__':
    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Run experiment
    run_exp1(config)