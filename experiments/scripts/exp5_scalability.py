#!/usr/bin/env python3
"""
Exp-5: Scalability and Ablation
Studying how AutoConfig scales with graph size and the impact of key components.

Ablation experiments:
- AutoConfig_noSPF: Without static program features
- AutoConfig_noSGF: Without symbolic graph-parameterized features
- AutoConfig_noBk: Without backup mechanism
"""

import os
import sys
import numpy as np
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "workloads"))

from autoconfig import CostPredictor, FeatureManager
from autoconfig.models import BayesianExecutionTimeModel
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


class AutoConfigAblation:
    """
    AutoConfig with ablated components for experimentation.
    """

    def __init__(
        self,
        variant: str = "full",
        use_backup: bool = True,
        evaluation_tau: float = 0.1
    ):
        """
        Initialize AutoConfig with ablation.

        Args:
            variant: One of 'full', 'noSPF', 'noSGF', 'raw'
            use_backup: Whether to use inspection-and-adjustment (backup) mechanism
            evaluation_tau: Sampling ratio for evaluation graph G'
        """
        self.variant = variant
        self.use_backup = use_backup
        self.evaluation_tau = evaluation_tau

        # Create custom feature manager based on variant
        self.feature_manager = self._create_feature_manager(variant)
        self.model = BayesianExecutionTimeModel()

    def _create_feature_manager(self, variant: str) -> FeatureManager:
        """Create feature manager with ablated components."""
        manager = FeatureManager()

        if variant == "noSPF":
            manager.static_extractor = None
        elif variant == "noSGF":
            manager.symbolic_extractor = None
        elif variant == "raw":
            manager.static_extractor = None
            manager.symbolic_extractor = None

        return manager

    def train(
        self,
        queries: List[str],
        graphs: List,
        configs: List[Dict],
        execution_times: np.ndarray,
        verbose: bool = False
    ):
        """Train the model."""
        features_list = []
        for i in range(len(queries)):
            features = self.feature_manager.extract_all(queries[i], graphs[i], configs[i])
            features_list.append(features)

        features_matrix = np.vstack(features_list)
        self.model.fit(features_matrix, execution_times, verbose=verbose)

    def recommend(
        self,
        query: str,
        graph: any,
        config_space: Dict[str, Any],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Recommend configuration(s) for given query and graph.

        Args:
            query: Query code
            graph: Data graph
            config_space: Configuration space definition
            top_n: Number of top recommendations to return

        Returns:
            List of recommended configurations with predicted costs
        """
        # Step 1: Generate candidate configurations
        candidates = self._generate_candidates(config_space, n_candidates=100)

        # Step 2: Predict costs for all candidates
        predictions = []
        for config in candidates:
            features = self.feature_manager.extract_all(query, graph, config)
            features = features.reshape(1, -1)
            pred_cost = self.model.predict(features)[0]

            predictions.append({
                'config': config,
                'predicted_cost': pred_cost
            })

        # Step 3: Sort by predicted cost
        predictions.sort(key=lambda x: x['predicted_cost'])

        # Step 4: Backup mechanism (inspection-and-adjustment)
        if self.use_backup:
            predictions = self._apply_backup_mechanism(
                query, graph, predictions, config_space
            )

        # Step 5: Return top N
        return predictions[:top_n]

    def _generate_candidates(
        self,
        config_space: Dict[str, Any],
        n_candidates: int = 100
    ) -> List[Dict[str, Any]]:
        """Generate candidate configurations using LHS sampling."""
        candidates = []

        np.random.seed(42)

        for i in range(n_candidates):
            candidate = {
                'k': np.random.randint(1, 65),
                'cpu_cores': np.random.choice([8, 16, 32, 64, 128, 256]),
                'memory_gb': np.random.choice([16, 32, 64, 128, 256, 512, 1024]),
                'num_gpus': np.random.choice([0, 1, 2, 4, 8])
            }
            candidates.append(candidate)

        return candidates

    def _apply_backup_mechanism(
        self,
        query: str,
        graph: any,
        predictions: List[Dict],
        config_space: Dict[str, Any],
        top_k: int = 3
    ) -> List[Dict]:
        """
        Apply inspection-and-adjustment (backup) mechanism.

        Evaluates top-k candidates on sampled subgraph G' and adjusts ranking.
        """
        if not self.use_backup:
            return predictions

        # Sample evaluation graph G'
        G_prime = self._sample_evaluation_graph(graph, self.evaluation_tau)

        # Evaluate top-k candidates on G'
        evaluated_costs = []
        for i, pred in enumerate(predictions[:top_k]):
            config = pred['config']

            # Simulate execution on G' (using synthetic cost)
            exec_cost = _compute_exec_time("AutoConfig", G_prime, config, noise_level=0.05)

            adjusted_cost = 0.7 * pred['predicted_cost'] + 0.3 * exec_cost

            evaluated_costs.append(adjusted_cost)

        # Adjust predictions based on G' evaluation
        for i, pred in enumerate(predictions[:top_k]):
            pred['adjusted_cost'] = evaluated_costs[i]
            pred['evaluated_on_G_prime'] = exec_cost

        # Re-sort based on adjusted cost
        for pred in predictions[:top_k]:
            pred['final_cost'] = pred.get('adjusted_cost', pred['predicted_cost'])

        predictions.sort(key=lambda x: x['final_cost'])

        return predictions

    def _sample_evaluation_graph(self, graph: any, tau: float) -> any:
        """
        Sample evaluation graph G' as subgraph of G.

        Args:
            graph: Original graph G
            tau: Sampling ratio (|V(G')| / |V(G)|)

        Returns:
            Sampled subgraph G'
        """
        import networkx as nx

        n_orig = graph.number_of_nodes()
        n_sampled = max(1, int(n_orig * tau))

        # Randomly sample nodes
        sampled_nodes = np.random.choice(
            list(graph.nodes()), n_sampled, replace=False
        )

        # Create subgraph
        G_prime = graph.subgraph(sampled_nodes).copy()

        return G_prime


def evaluate_graph_size_scaling(
    workload_name: str,
    query_code: str,
    graph_sizes: List[int],
    variants: List[str],
    config_space: Dict
) -> Dict[str, Dict[str, List[float]]]:
    """
    Evaluate how method scales with graph size.

    Tests on graphs of increasing size.
    """
    results = {variant: {'runtime': [], 'selection_time': []} for variant in variants}

    for graph_size in graph_sizes:
        print(f"  Testing with graph size: {graph_size} nodes")

        import networkx as nx
        graph = nx.barabasi_albert_graph(graph_size, 2)

        for variant in variants:
            # Initialize AutoConfig variant
            autoconf = AutoConfigAblation(
                variant=variant,
                use_backup=True,
                evaluation_tau=0.1
            )

            # Train with some data
            queries, train_graphs, train_configs, train_times = generate_training_data(
                200, workload_name
            )
            autoconf.train(queries, train_graphs, train_configs, train_times)

            # Measure recommendation time
            import time
            start = time.time()
            recommendations = autoconf.recommend(query_code, graph, config_space, top_n=5)
            selection_time = (time.time() - start) * 1000  # ms

            # Measure actual runtime using top recommendation
            top_config = recommendations[0]['config']
            actual_runtime = _compute_exec_time(workload_name, graph, top_config)

            results[variant]['runtime'].append(actual_runtime)
            results[variant]['selection_time'].append(selection_time)

    return results


def evaluate_cluster_size_scaling(
    workload_name: str,
    query_code: str,
    graph: any,
    cluster_sizes: List[int],
    variants: List[str],
    config_space: Dict
) -> Dict[str, Dict[str, List[float]]]:
    """
    Evaluate how method scales with cluster size (k, CPU cores, GPUs).

    Tests on larger resource configurations.
    """
    results = {variant: {'runtime': [], 'scalability': []} for variant in variants}

    for cluster_size in cluster_sizes:
        print(f"  Testing with cluster size: {cluster_size} cores")

        # Create config representing larger cluster
        base_config = {
            'k': min(cluster_size // 8, 64),
            'cpu_cores': cluster_size,
            'memory_gb': cluster_size * 2,
            'num_gpus': max(1, cluster_size // 64)
        }

        for variant in variants:
            # Initialize AutoConfig variant
            autoconf = AutoConfigAblation(
                variant=variant,
                use_backup=True,
                evaluation_tau=0.1
            )

            # Train
            queries, train_graphs, train_configs, train_times = generate_training_data(
                200, workload_name
            )
            autoconf.train(queries, train_graphs, train_configs, train_times)

            # Recommend and execute
            recommendations = autoconf.recommend(query_code, graph, config_space, top_n=5)
            top_config = recommendations[0]['config']

            # Adjust to requested cluster size
            top_config['cpu_cores'] = base_config['cpu_cores']
            top_config['memory_gb'] = base_config['memory_gb']

            actual_runtime = _compute_exec_time(workload_name, graph, top_config)

            # Compute scalability (speedup compared to 1 core)
            baseline_runtime = _compute_exec_time(workload_name, graph, {
                'k': 1, 'cpu_cores': 1, 'memory_gb': 16, 'num_gpus': 0
            })

            scalability = baseline_runtime / actual_runtime

            results[variant]['runtime'].append(actual_runtime)
            results[variant]['scalability'].append(scalability)

    return results


def evaluate_component_ablation(
    workload_name: str,
    query_code: str,
    graphs: List,
    config_space: Dict
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate impact of each component via ablation.

    Compares: full vs noSPF vs noSGF vs noBk
    """
    variants = ['full', 'noSPF', 'noSGF', 'raw']

    results = {}

    for variant in variants:
        print(f"  Testing variant: {variant}")

        # Initialize without backup for ablation
        autoconf = AutoConfigAblation(
            variant=variant,
            use_backup=True if variant == 'full' else False,
            evaluation_tau=0.1
        )

        # Also test full without backup
        if variant == 'full':
            autoconf_no_bkup = AutoConfigAblation(
                variant='full',
                use_backup=False,
                evaluation_tau=0.1
            )

        # Train
        queries, train_graphs, train_configs, train_times = generate_training_data(
            200, workload_name
        )
        autoconf.train(queries, train_graphs, train_configs, train_times)

        if variant == 'full':
            autoconf_no_bkup.train(queries, train_graphs, train_configs, train_times)

        # Test on multiple graphs
        costs = []
        costs_no_bkup = []

        for graph in graphs:
            recommendations = autoconf.recommend(query_code, graph, config_space, top_n=5)
            top_config = recommendations[0]['config']
            cost = _compute_exec_time(workload_name, graph, top_config)
            costs.append(cost)

            if variant == 'full':
                recs = autoconf_no_bkup.recommend(query_code, graph, config_space, top_n=5)
                top_c = recs[0]['config']
                cost_no_bkup = _compute_exec_time(workload_name, graph, top_c)
                costs_no_bkup.append(cost_no_bkup)

        results[variant] = {
            'mean_cost': float(np.mean(costs)),
            'std_cost': float(np.std(costs)),
        }

        if variant == 'full':
            results['noBk'] = {
                'mean_cost': float(np.mean(costs_no_bkup)),
                'std_cost': float(np.std(costs_no_bkup)),
            }

    # Compute degradation percentages
    full_cost = results['full']['mean_cost']

    degradation = {}
    for variant in ['noSPF', 'noSGF', 'raw', 'noBk']:
        if variant in results:
            degradation[f'{variant}_degradation'] = (
                (results[variant]['mean_cost'] / full_cost) - 1
            ) * 100

    results['degradation'] = degradation

    return degradation


def run_exp5(config: Dict[str, Any]):
    """
    Run complete Exp-5: Scalability and Ablation.
    """
    print(f"\n{'#'*70}")
    print(f"# Exp-5: Scalability and Ablation")
    print(f"{'#'*70}")

    output_dir = Path(config['global']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    workloads = {
        'WCC': query_wcc,
        'PR': query_pr,
        'BFS': query_bfs,
    }

    # Config space
    config_space = {
        'k_range': config['configuration_space']['k_range'],
        'resources': config['configuration_space']['resources']
    }

    # Variants to test
    variants = ['full', 'noSPF', 'noSGF', 'raw']

    # ========== Part 1: Graph size scaling ==========
    print(f"\n{'='*70}")
    print(f"Part 1: Graph size scaling")
    print(f"{'='*70}")

    graph_sizes = [1000, 5000, 10000, 50000, 100000]

    graph_scaling_results = {}
    for wl_name, query_code in workloads.items():
        print(f"\nWorkload: {wl_name}")

        results = evaluate_graph_size_scaling(
            wl_name, query_code, graph_sizes, variants, config_space
        )
        graph_scaling_results[wl_name] = results

    exp5_graph_scaling_output = output_dir / "exp5_graph_scaling.json"
    with open(exp5_graph_scaling_output, 'w') as f:
        json.dump(graph_scaling_results, f, indent=2)
    print(f"  Saved to {exp5_graph_scaling_output}")

    # ========== Part 2: Cluster size scaling ==========
    print(f"\n{'='*70}")
    print(f"Part 2: Cluster size scaling")
    print(f"{'='*70}")

    cluster_sizes = [16, 32, 64, 128, 256]

    import networkx as nx
    test_graph = nx.barabasi_albert_graph(10000, 2)

    cluster_scaling_results = {}
    for wl_name, query_code in workloads.items():
        print(f"\nWorkload: {wl_name}")

        results = evaluate_cluster_size_scaling(
            wl_name, query_code, test_graph, cluster_sizes, variants, config_space
        )
        cluster_scaling_results[wl_name] = results

    exp5_cluster_scaling_output = output_dir / "exp5_cluster_scaling.json"
    with open(exp5_cluster_scaling_output, 'w') as f:
        json.dump(cluster_scaling_results, f, indent=2)
    print(f"  Saved to {exp5_cluster_scaling_output}")

    # ========== Part 3: Component ablation ==========
    print(f"\n{'='*70}")
    print(f"Part 3: Component ablation")
    print(f"{'='*70}")

    import networkx as nx
    test_graphs = [nx.barabasi_albert_graph(5000, 2) for _ in range(5)]

    ablation_results = {}
    for wl_name, query_code in workloads.items():
        print(f"\nWorkload: {wl_name}")

        results = evaluate_component_ablation(
            wl_name, query_code, test_graphs, config_space
        )
        ablation_results[wl_name] = results

    exp5_ablation_output = output_dir / "exp5_ablation.json"
    with open(exp5_ablation_output, 'w') as f:
        json.dump(ablation_results, f, indent=2)
    print(f"  Saved to {exp5_ablation_output}")

    # Summary
    print(f"\n{'='*70}")
    print(f"Exp-5 Summary")
    print(f"{'='*70}")

    summary = {
        'graph_scaling': graph_scaling_results,
        'cluster_scaling': cluster_scaling_results,
        'ablation': ablation_results,
    }

    summary_path = output_dir / "exp5_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == '__main__':
    config_path = Path(__file__).parent.parent / "config.yaml"

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    run_exp5(config)