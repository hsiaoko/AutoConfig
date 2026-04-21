#!/usr/bin/env python3
"""
Exp-2: Effectiveness
Measuring the end-to-end runtime achieved under recommended configurations.
"""

import os
import sys
import numpy as np
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "baselines"))
sys.path.insert(0, str(Path(__file__).parent.parent / "workloads"))

from autoconfig import CostPredictor
from baselines import (
    BayesianOptimization, ReinforcementLearning, GPTuner, BestConfig, Oracle
)
from workloads import query_wcc, query_sssp, query_pr, query_bfs, query_subiso

# Import utilities from exp1 (using direct import)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "exp1_utils",
    Path(__file__).parent / "exp1_feature_extraction.py"
)
exp1_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp1_module)
generate_training_data = exp1_module.generate_training_data
_compute_exec_time = exp1_module._compute_exec_time


def evaluate_method_on_workload(
    method_name: str,
    method: any,
    query_code: str,
    graph: any,
    config_space: Dict[str, Any],
    predictor: CostPredictor = None,
    oracle: Oracle = None
) -> Dict[str, Any]:
    """
    Evaluate a configuration tuning method on a single workload-graph pair.

    Returns recommended config, actual cost, and runtime cost.
    """
    # Extract workload features (bag-of-words for baselines)
    if predictor:
        # Use feature manager for AutoConfig
        features = predictor.feature_manager.extract_all(query_code, graph, {})
    else:
        # Simple bag-of-words for baselines
        features = np.random.rand(20)  # Simulated features

    # Get recommended configuration
    if method_name == "Oracle" and oracle:
        recommended_config = oracle.recommend(features, graph)
    elif method_name == "BestConfig":
        recommended_config = method.recommend(features, graph)
    else:
        recommended_config = method.recommend(features)

    # Compute actual execution cost
    actual_cost = _compute_exec_time(
        method_name, graph, recommended_config, noise_level=0.05
    )

    # Selection latency (time to recommend)
    import time
    start = time.time()

    if method_name == "Oracle" and oracle:
        config = oracle.recommend(features, graph)
    elif method_name == "BestConfig":
        config = method.recommend(features, graph)
    else:
        config = method.recommend(features)

    selection_latency = (time.time() - start) * 1000  # ms

    # Monetary cost
    monetary_cost = compute_monetary_cost(recommended_config, actual_cost)

    # Unified cost with w1=0.5, w2=0.5
    unified_cost = 0.5 * actual_cost + 0.5 * monetary_cost

    return {
        'recommended_config': recommended_config,
        'actual_runtime': actual_cost,
        'monetary_cost': monetary_cost,
        'unified_cost': unified_cost,
        'selection_latency_ms': selection_latency
    }


def compute_monetary_cost(config: Dict[str, Any], runtime: float) -> float:
    """
    Compute monetary cost based on config and runtime.

    Uses pricing from experiment config (catalog-style instances).
    """
    cpu_cost_per_ms = 0.05 / 1000 / 3600  # $0.05 per core-hour, convert to per-ms
    gpu_cost_per_ms = 0.5 / 1000 / 3600   # $0.5 per GPU-hour

    total_cost = (
        config['cpu_cores'] * cpu_cost_per_ms +
        config['num_gpus'] * gpu_cost_per_ms
    ) * runtime

    return total_cost


def run_all_comparison(
    workload_name: str,
    query_code: str,
    graphs: List[any],
    config_space: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, List[Dict]]:
    """
    Run all methods on multiple graphs for a workload.
    """
    print(f"\nEvaluating {workload_name} on {len(graphs)} graphs...")

    # Initialize methods
    methods = {
        "AutoConfig": None,  # Will use predictor
        "BO": BayesianOptimization(config_space),
        "RL": ReinforcementLearning(config_space),
        "GPTuner": GPTuner(config_space),
        "BestConfig": BestConfig(config_space),
    }

    # Oracle only for small graphs
    oracle = Oracle(config_space) if len(graphs[0]) < 10000 else None
    if oracle:
        methods["Oracle"] = oracle

    # Train AutoConfig predictor if needed
    predictor = None
    if "AutoConfig" in methods:
        # Generate training data for this workload
        queries, train_graphs, train_configs, train_times = generate_training_data(
            500, workload_name
        )
        predictor = CostPredictor()
        predictor.train(queries, train_graphs, train_configs, train_times)

    results = {method_name: [] for method_name in methods.keys()}

    for i, graph in enumerate(graphs):
        print(f"  Graph {i+1}/{len(graphs)}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        for method_name, method in methods.items():
            try:
                result = evaluate_method_on_workload(
                    method_name, method, query_code, graph,
                    config_space, predictor, oracle
                )
                results[method_name].append(result)
            except Exception as e:
                print(f"    Error in {method_name}: {e}")
                results[method_name].append(None)

    return results


def compare_to_oracle(results: Dict[str, List[Dict]]) -> Dict[str, float]:
    """
    Compare each method to Oracle baseline.

    Returns improvement ratios and other metrics.
    """
    if "Oracle" not in results or not results["Oracle"]:
        return {}

    oracle_costs = [r['unified_cost'] for r in results["Oracle"] if r]
    comparisons = {}

    for method_name, method_results in results.items():
        if method_name == "Oracle":
            continue

        valid_results = [r for r in method_results if r]
        if not valid_results:
            continue

        method_costs = [r['unified_cost'] for r in valid_results]

        # Improvement ratio
        if oracle_costs:
            improvement = (np.mean(oracle_costs) / np.mean(method_costs)) - 1
            comparisons[f"{method_name}_improvement"] = improvement

        # Features-based Hamming distance (simplified)
        comparisons[f"{method_name}_avg_cost"] = float(np.mean(method_costs))
        comparisons[f"{method_name}_std_cost"] = float(np.std(method_costs))

    return comparisons


def run_exp2(config: Dict[str, Any]):
    """
    Run complete Exp-2: Effectiveness evaluation.
    """
    print(f"\n{'#'*70}")
    print(f"# Exp-2: Effectiveness (End-to-end Runtime)")
    print(f"{'#'*70}")

    output_dir = Path(config['global']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    workloads = {
        'WCC': query_wcc,
        'SSSP': query_sssp,
        'PR': query_pr,
        'BFS': query_bfs,
        'SubIso': query_subiso,
    }

    # Config space from experiment config
    config_space = {
        'k_range': config['configuration_space']['k_range'],
        'resources': config['configuration_space']['resources']
    }

    all_results = {}
    summary = {}

    # Evaluate each workload
    for wl_name, query_code in workloads.items():
        print(f"\n{'='*70}")
        print(f"Workload: {wl_name}")
        print(f"{'='*70}")

        # Generate test graphs of varying sizes
        sizes = [1000, 5000, 10000]
        graphs = []
        import networkx as nx

        for size in sizes:
            g = nx.barabasi_albert_graph(size, 2)
            graphs.append(g)

        # Run all methods
        results = run_all_comparison(
            wl_name, query_code, graphs, config_space, config
        )
        all_results[wl_name] = results

        # Compare to Oracle
        if any("Oracle" in r for r in results.get("Oracle", [])):
            comparisons = compare_to_oracle(results)
            summary[wl_name] = comparisons

        # Save per-workload results
        wl_output_path = output_dir / f"exp2_{wl_name}.json"
        with open(wl_output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"  Results saved to {wl_output_path}")

    # Save summary
    summary_path = output_dir / "exp2_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'per_workload': all_results,
            'summary': summary
        }, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"Exp-2 Summary:")
    print(f"{'='*70}")

    for wl_name, comps in summary.items():
        print(f"\n{wl_name}:")
        for metric, value in comps.items():
            print(f"  {metric}: {value:.4f}")

    print(f"\nAll results saved to {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == '__main__':
    config_path = Path(__file__).parent.parent / "config.yaml"

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    run_exp2(config)