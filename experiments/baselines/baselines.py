"""
Baseline configuration tuning methods.

Adapted for graph workloads using bag-of-words feature representation.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import networkx as nx


class BaselineMethod(ABC):
    """Abstract base class for configuration tuning baselines."""

    def __init__(self, config_space: Dict[str, Any], cost_model=None):
        """
        Initialize baseline method.

        Args:
            config_space: Dictionary defining configuration space
            cost_model: Cost estimation model (Oracle has access to true costs)
        """
        self.config_space = config_space
        self.cost_model = cost_model
        self.history = []

    @abstractmethod
    def recommend(self, workload_features: np.ndarray) -> Dict[str, Any]:
        """Recommend a configuration for given workload."""
        pass

    def evaluate_config(
        self,
        config: Dict[str, Any],
        workload_features: np.ndarray,
        graph: nx.Graph,
        use_true_cost: bool = False
    ) -> float:
        """
        Evaluate a configuration (simulated execution).

        Args:
            config: Configuration to evaluate
            workload_features: Feature vector of the workload
            graph: Data graph
            use_true_cost: If True, use oracle costs; otherwise use cost_model estimate

        Returns:
            Cost (runtime + monetary)
        """
        if use_true_cost and cost_model_name := self.cost_model:
            return self.cost_model.true_cost(config, graph)
        elif self.cost_model:
            return self.cost_model.predicted_cost(config, graph)
        else:
            # Fallback: simple synthetic cost
            return self._synthetic_cost(config, graph, workload_features)

    def _synthetic_cost(
        self,
        config: Dict[str, Any],
        graph: nx.Graph,
        features: np.ndarray
    ) -> float:
        """Synthetic cost calculation for simulation."""
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()

        # Graph size cost
        graph_cost = n_nodes * 0.01 + n_edges * 0.002

        # Configuration cost (inversely related to resources)
        k = config.get('k', 1)
        cpu = config.get('cpu_cores', 1)
        gpu = config.get('num_gpus', 0)

        resource_cost = 64.0 / cpu + 4.0 / (k + 1)
        resource_cost *= (1.0 if gpu == 0 else 0.5 / gpu)

        # Workload complexity from first feature dimension
        workload_cost = features[0] if len(features) > 0 else 1.0

        total_cost = graph_cost * resource_cost * workload_cost

        return total_cost


class BayesianOptimization(BaselineMethod):
    """
    Bayesian Optimization for configuration tuning.

    Uses Gaussian Process surrogate model to learn cost landscape.
    """

    def __init__(self, config_space: Dict[str, Any], cost_model=None):
        super().__init__(config_space, cost_model)

    def recommend(self, workload_features: np.ndarray) -> Dict[str, Any]:
        """
        Recommend configuration using BO exploration-exploitation.

        For fair comparison, uses bag-of-words features extracted from workload.
        """
        # In a real implementation, this would:
        # 1. Fit GP surrogate model to historical logs
        # 2. Optimize acquisition function (e.g, EI, UCB)
        # 3. Return top recommended config

        # Simplified implementation: suggest based on workload features
        # Heuristic: larger graphs need more resources
        if len(workload_features) > 1:
            resource_level = int(workload_features[1] * 10)  # Size indicator
            resource_level = max(1, min(5, resource_level))
        else:
            resource_level = 2

        k_values = {1: 1, 2: 4, 3: 16, 4: 32, 5: 64}
        cpu_values = {1: 8, 2: 16, 3: 32, 4: 64, 5: 128}
        gpu_values = {1: 0, 2: 0, 3: 1, 4: 2, 5: 4}

        return {
            'k': k_values[resource_level],
            'cpu_cores': cpu_values[resource_level],
            'memory_gb': cpu_values[resource_level] * 2,
            'num_gpus': gpu_values[resource_level]
        }


class ReinforcementLearning(BaselineMethod):
    """
    Reinforcement Learning for configuration tuning.

    Trains policy offline from execution logs.
    """

    def __init__(self, config_space: Dict[str, Any], cost_model=None):
        super().__init__(config_space, cost_model)
        self.policy = None  # Would be trained offline

    def recommend(self, workload_features: np.ndarray) -> Dict[str, Any]:
        """
        Recommend configuration using learned policy.

        Policy directly maps workload features to configuration decisions.
        """
        # Simplified: map features to actions using heuristics
        # In practice, this would be a trained neural network policy

        if len(workload_features) < 2:
            feature_code = 2
        else:
            # Create feature code from first two features
            feature_code = int(workload_features[0]) + int(workload_features[1] * 3)
            feature_code = min(feature_code, 10)

        # Lookup table for policy (simplified)
        policy_table = [
            {'k': 1, 'cpu_cores': 8, 'memory_gb': 16, 'num_gpus': 0},  # 0
            {'k': 2, 'cpu_cores': 16, 'memory_gb': 32, 'num_gpus': 0},  # 1
            {'k': 4, 'cpu_cores': 32, 'memory_gb': 64, 'num_gpus': 0},  # 2
            {'k': 8, 'cpu_cores': 32, 'memory_gb': 128, 'num_gpus': 1},  # 3
            {'k': 8, 'cpu_cores': 64, 'memory_gb': 128, 'num_gpus': 1},  # 4
            {'k': 16, 'cpu_cores': 64, 'memory_gb': 256, 'num_gpus': 2},  # 5
            {'k': 16, 'cpu_cores': 64, 'memory_gb': 256, 'num_gpus': 2},  # 6
            {'k': 32, 'cpu_cores': 128, 'memory_gb': 512, 'num_gpus': 4},  # 7
            {'k': 32, 'cpu_cores': 128, 'memory_gb': 512, 'num_gpus': 4},  # 8
            {'k': 64, 'cpu_cores': 256, 'memory_gb': 512, 'num_gpus': 8},  # 9
            {'k': 64, 'cpu_cores': 256, 'memory_gb': 1024, 'num_gpus': 8},  # 10
        ]

        return policy_table[feature_code]


class GPTuner(BaselineMethod):
    """
    GPTuner: LLM-enhanced BO.

    Extends BO by injecting domain knowledge via large language models.
    """

    def __init__(self, config_space: Dict[str, Any], cost_model=None):
        super().__init__(config_space, cost_model)

    def recommend(self, workload_features: np.ndarray) -> Dict[str, Any]:
        """
        Recommend configuration with LLM-informed BO.

        Uses both BO surrogate model and LLM-generated domain insights.
        """
        # Vanilla BO recommendation
        bo_rec = BayesianOptimization(self.config_space, self.cost_model).recommend(
            workload_features
        )

        # Apply LLM-informed adjustments (simplified)
        # In practice, LLM would analyze workload code and suggest adjustments

        # Example heuristic: if workload has recursion, add resources
        if len(workload_features) > 2 and workload_features[2] > 0.5:
            bo_rec['cpu_cores'] = int(bo_rec['cpu_cores'] * 1.5)
            bo_rec['memory_gb'] = int(bo_rec['memory_gb'] * 1.5)

        return bo_rec


class BestConfig(BaselineMethod):
    """
    BestConfig: Grid-style search heuristic.

    Evaluates configurations one by one using sampled datasets.
    """

    def __init__(self, config_space: Dict[str, Any], cost_model=None):
        super().__init__(config_space, cost_model)
        self.evaluation_cache = {}

    def recommend(
        self,
        workload_features: np.ndarray,
        graph: Optional[nx.Graph] = None,
        max_evaluations: int = 10
    ) -> Dict[str, Any]:
        """
        Recommend configuration by evaluating sampled configs.

        Performs grid-style search, evaluating configs sequentially.
        """
        # Generate candidate configurations
        candidates = self._generate_candidates(max_evaluations)

        if graph is None:
            # If no graph provided, return first candidate
            return candidates[0]

        # Evaluate each candidate
        best_config = None
        best_cost = float('inf')

        for i, config in enumerate(candidates):
            cost = self.evaluate_config(config, workload_features, graph)
            self.history.append({'config': config, 'cost': cost})

            if cost < best_cost:
                best_cost = cost
                best_config = config.copy()

        return best_config

    def _generate_candidates(self, n: int) -> List[Dict[str, Any]]:
        """Generate candidate configurations using grid search."""
        candidates = []

        # Coarse grid on key dimensions
        k_options = [1, 4, 8, 16, 32, 64]
        cpu_options = [8, 16, 32, 64, 128, 256]
        gpu_options = [0, 1, 2, 4, 8]

        # Total combinations = k_options.size * cpu_options.size * gpu_options.size
        # We sample n candidates uniformly

        np.random.seed(42)
        total_combos = len(k_options) * len(cpu_options) * len(gpu_options)
        step = max(1, total_combos // n)

        idx = 0
        for i, k in enumerate(k_options):
            for j, cpu in enumerate(cpu_options):
                for _, gpu in enumerate(gpu_options):
                    if idx % step == 0 and len(candidates) < n:
                        candidates.append({
                            'k': int(k),
                            'cpu_cores': int(cpu),
                            'memory_gb': int(cpu * 2),
                            'num_gpus': int(gpu)
                        })
                    idx += 1

        return candidates


class Oracle(BaselineMethod):
    """
    Oracle: Exhaustive search for provably optimal configuration.

    Serves as upper bound, feasible for small/medium graphs only.
    """

    def __init__(
        self,
        config_space: Dict[str, Any],
        cost_model,
        exhaustive_search: bool = True
    ):
        super().__init__(config_space, cost_model)
        self.exhaustive_search = exhaustive_search

    def recommend(
        self,
        workload_features: np.ndarray,
        graph: nx.Graph
    ) -> Dict[str, Any]:
        """
        Find provably optimal configuration via exhaustive search.

        Only feasible for small/medium graphs.
        """
        if not self.exhaustive_search:
            # For large graphs, use approximate search
            return self._approximate_search(workload_features, graph)

        # Generate all possible configurations
        all_configs = self._generate_all_configs()

        # Evaluate each configuration
        best_config = None
        best_cost = float('inf')

        for config in all_configs:
            cost = self.evaluate_config(
                config, workload_features, graph, use_true_cost=True
            )

            if cost < best_cost:
                best_cost = cost
                best_config = config.copy()

        return best_config

    def _generate_all_configs(self) -> List[Dict[str, Any]]:
        """Generate all possible configurations."""
        configs = []

        k_range = self.config_space.get('k_range', [1, 64])
        cpu_range = self.config_space.get('resources', [{}])[0].get('cores_range', [1, 256])
        gpu_range = self.config_space.get('resources', [{}])[2].get('count_range', [0, 8])

        # For exhaustive search, use coarse granularity
        k_values = [1, 4, 8, 16, 32, 64]
        cpu_values = [8, 16, 32, 64, 128, 256]
        gpu_values = [0, 1, 2, 4, 8]

        for k in k_values:
            for cpu in cpu_values:
                for gpu in gpu_values:
                    configs.append({
                        'k': k,
                        'cpu_cores': cpu,
                        'memory_gb': cpu * 2,
                        'num_gpus': gpu
                    })

        return configs

    def _approximate_search(
        self,
        workload_features: np.ndarray,
        graph: nx.Graph
    ) -> Dict[str, Any]:
        """
        Approximate search for large graphs.

        Samples configurations and returns best found.
        """
        # Sample configurations
        n_samples = 50
        configs = []

        np.random.seed(42)
        for i in range(n_samples):
            configs.append({
                'k': np.random.choice([1, 4, 8, 16, 32, 64]),
                'cpu_cores': np.random.choice([8, 16, 32, 64, 128, 256]),
                'memory_gb': np.random.choice([16, 32, 64, 128, 256, 512, 1024]),
                'num_gpus': np.random.choice([0, 1, 2, 4, 8])
            })

        # Evaluate sampled configs
        best_config = None
        best_cost = float('inf')

        for config in configs:
            cost = self.evaluate_config(
                config, workload_features, graph, use_true_cost=True
            )

            if cost < best_cost:
                best_cost = cost
                best_config = config.copy()

        return best_config