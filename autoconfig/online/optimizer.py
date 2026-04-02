"""
Configuration Optimizer

Ranks configurations by predicted cost and refines via perturbation.
Returns diverse top-K configurations.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Set
from copy import deepcopy

from .cost_predictor import CostPredictor
from ..offline.data_generator import DataGenerator


class ConfigurationOptimizer:
    """
    Optimizer for finding best configurations.

    Process:
    1. Rank configurations by predicted execution cost
    2. Select top-K candidates
    3. Apply random perturbations to explore neighborhood
    4. Return diverse top-K configurations (different resource configs)
    """

    def __init__(
        self,
        cost_predictor: Optional[CostPredictor] = None,
        top_k: int = 3,
        perturbation_range: int = 2,
        num_perturbations: int = 5
    ):
        """
        Initialize optimizer.

        Args:
            cost_predictor: Cost predictor instance
            top_k: Number of top candidates to return
            perturbation_range: Range for core count perturbation (+/- n)
            num_perturbations: Number of perturbations per candidate
        """
        self.cost_predictor = cost_predictor or CostPredictor()
        self.top_k = top_k
        self.perturbation_range = perturbation_range
        self.num_perturbations = num_perturbations
        self.data_generator = DataGenerator()

    def rank_configurations(
        self,
        query_complexity: Dict[str, int],
        graph_features: Dict[str, Any],
        candidate_configs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rank configurations by predicted execution cost.

        Args:
            query_complexity: Query complexity dictionary
            graph_features: Graph features dictionary
            candidate_configs: List of candidate configurations

        Returns:
            Sorted list of configurations with predictions
        """
        ranked = []

        for config in candidate_configs:
            resource = config['resource']

            try:
                predictions = self.cost_predictor.predict_both(
                    query_complexity, graph_features, resource
                )
            except (ValueError, KeyError):
                # Models not available, use heuristic
                predictions = self._heuristic_cost(resource, graph_features)

            # Handle both model predictions and heuristic results
            pred_time = predictions.get('execution_time_ms') or predictions.get('predicted_time', 0)
            pred_cost = predictions.get('execution_cost') or predictions.get('predicted_cost', 1.0)

            ranked.append({
                'config': config,
                'predicted_time': pred_time,
                'predicted_cost': pred_cost if pred_cost > 0 else 1.0,
                'time_lower': predictions.get('execution_time_lower', 0),
                'time_upper': predictions.get('execution_time_upper', 0),
            })

        # Sort by cost (ascending)
        ranked.sort(key=lambda x: x['predicted_cost'])

        return ranked

    def _heuristic_cost(
        self,
        resource: Dict[str, Any],
        graph_features: Dict[str, Any]
    ) -> Dict[str, float]:
        """Fallback heuristic when models are not available."""
        V = graph_features.get('num_vertices', 1000)
        E = graph_features.get('num_edges', 5000)
        cpu = resource.get('cpu_cores', 8)
        gpu = resource.get('num_gpus', 0)

        # Simple heuristic
        base_time = (V * 0.01 + E * 0.001) * (32 / max(cpu, 1))
        if gpu > 0:
            base_time *= 0.5

        cost = base_time * cpu * 0.05  # Arbitrary cost units

        return {
            'execution_time_ms': base_time,
            'execution_cost': cost,
        }

    def _config_signature(self, resource: Dict[str, Any]) -> str:
        """
        Generate a unique signature for a resource configuration.
        
        Used to detect duplicate configurations.
        """
        return f"CPU:{resource.get('cpu_cores', 0)}_MEM:{resource.get('memory_gb', 0)}_GPU:{resource.get('num_gpus', 0)}"

    def perturb_config(
        self,
        config: Dict[str, Any],
        delta: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Apply perturbation to a configuration.

        Args:
            config: Base configuration
            delta: Specific delta to apply (or None for random)

        Returns:
            Perturbed configuration
        """
        perturbed = deepcopy(config)
        resource = perturbed['resource']

        # Perturb CPU cores
        if delta is None:
            delta = np.random.randint(-self.perturbation_range, self.perturbation_range + 1)
        
        new_cores = max(1, resource['cpu_cores'] + delta * 4)
        resource['cpu_cores'] = new_cores

        # Update config ID to reflect perturbation
        original_id = config.get('config_id', 'config')
        # Remove any existing perturbation suffix
        base_id = original_id.split('_pert_')[0]
        perturbed['config_id'] = f"{base_id}_pert_{delta:+d}"
        perturbed['perturbation_delta'] = delta

        return perturbed

    def generate_perturbations(
        self,
        base_config: Dict[str, Any],
        query_complexity: Dict[str, int],
        graph_features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple perturbations of a base configuration.

        Ensures each perturbation has a unique CPU core count.

        Args:
            base_config: Base configuration
            query_complexity: Query complexity dictionary
            graph_features: Graph features dictionary

        Returns:
            List of perturbed configurations with predictions
        """
        perturbations = []
        seen_cores: Set[int] = {base_config['resource'].get('cpu_cores', 0)}

        for _ in range(self.num_perturbations):
            # Try to find a unique delta
            attempts = 0
            delta = 0
            
            while attempts < 10:
                delta = np.random.randint(-self.perturbation_range, self.perturbation_range + 1)
                if delta == 0:
                    continue  # Skip no-op perturbation
                    
                new_cores = base_config['resource'].get('cpu_cores', 0) + delta * 4
                if new_cores not in seen_cores and new_cores >= 1:
                    seen_cores.add(new_cores)
                    break
                attempts += 1
            
            if attempts >= 10:
                # Couldn't find unique delta, use what we have
                delta = np.random.choice([-1, 1])
            
            perturbed = self.perturb_config(base_config, delta)
            resource = perturbed['resource']

            try:
                predictions = self.cost_predictor.predict_both(
                    query_complexity, graph_features, resource
                )
            except (ValueError, KeyError):
                predictions = self._heuristic_cost(resource, graph_features)

            pred_time = predictions.get('execution_time_ms') or predictions.get('predicted_time', 0)
            pred_cost = predictions.get('execution_cost') or predictions.get('predicted_cost', 1.0)

            perturbations.append({
                'config': perturbed,
                'predicted_time': pred_time,
                'predicted_cost': pred_cost if pred_cost > 0 else 1.0,
                'parent_config': base_config.get('config_id', 'unknown'),
                'is_perturbed': True,
            })

        return perturbations

    def optimize(
        self,
        query_complexity: Dict[str, int],
        graph_features: Dict[str, Any],
        candidate_configs: Optional[List[Dict[str, Any]]] = None,
        return_all: bool = False
    ) -> Dict[str, Any]:
        """
        Full optimization pipeline.

        Returns diverse top-K configurations with different resource allocations.

        Args:
            query_complexity: Query complexity dictionary
            graph_features: Graph features dictionary
            candidate_configs: List of candidate configs or None for default
            return_all: Whether to return all explored configs

        Returns:
            Dictionary with best configurations
        """
        # Use default configs if not provided
        if candidate_configs is None:
            candidate_configs = self.data_generator.RESOURCE_CONFIGS
            candidate_configs = [{'config_id': f'default_{i}', 'resource': r, 'k': 1}
                                  for i, r in enumerate(candidate_configs)]

        # Step 1: Rank all configurations
        ranked = self.rank_configurations(
            query_complexity, graph_features, candidate_configs
        )

        # Step 2: Generate perturbations for top candidates
        all_explored = list(ranked)  # Start with original configs
        
        # Mark original configs as not perturbed
        for item in all_explored:
            item['is_perturbed'] = False
        
        # Generate perturbations for top-K original configs
        top_k_original = ranked[:min(self.top_k, len(ranked))]
        for ranked_config in top_k_original:
            perturbations = self.generate_perturbations(
                ranked_config['config'],
                query_complexity,
                graph_features
            )
            all_explored.extend(perturbations)

        # Step 3: Sort all by cost
        all_explored.sort(key=lambda x: x['predicted_cost'])

        # Step 4: Select diverse top-K (different CPU cores)
        diverse_top_k = []
        seen_signatures: Set[str] = set()

        for item in all_explored:
            sig = self._config_signature(item['config']['resource'])
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                diverse_top_k.append(item)
                
                if len(diverse_top_k) >= self.top_k:
                    break

        # If we don't have enough diverse configs, add remaining best ones
        if len(diverse_top_k) < self.top_k:
            for item in all_explored:
                if item not in diverse_top_k:
                    diverse_top_k.append(item)
                    if len(diverse_top_k) >= self.top_k:
                        break

        # Assign ranks
        for i, item in enumerate(diverse_top_k):
            item['rank'] = i + 1

        best = diverse_top_k[0] if diverse_top_k else None

        if return_all:
            return {
                'best_config': best,
                'diverse_top_k': diverse_top_k,
                'all_explored': all_explored,
            }

        return {
            'best_config': best,
            'diverse_top_k': diverse_top_k,
        }
