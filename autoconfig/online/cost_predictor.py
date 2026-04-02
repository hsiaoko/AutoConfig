"""
Cost Predictor for Online Inference

Predicts execution time and cost for given query + graph + config.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

from ..models.bayesian_models import BayesianTimeModel, BayesianCostModel
from ..feature_extractor import FeatureManager


class CostPredictor:
    """
    Online cost predictor using trained Bayesian models.
    
    Predicts:
    - Execution time (ms)
    - Execution cost (monetary units)
    - Uncertainty estimates
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize predictor with trained models.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = Path(model_dir) if model_dir else Path('data/models')
        self.feature_manager = FeatureManager()
        
        self.time_model = None
        self.cost_model = None
        
        # Try to load models
        self._try_load_models()
    
    def _try_load_models(self):
        """Load models if available."""
        time_path = self.model_dir / 'time_model.pkl'
        cost_path = self.model_dir / 'cost_model.pkl'
        
        if time_path.exists():
            self.time_model = BayesianTimeModel.load(str(time_path))
        if cost_path.exists():
            self.cost_model = BayesianCostModel.load(str(cost_path))
    
    def load_models(self, model_dir: str):
        """Explicitly load models from directory."""
        self.model_dir = Path(model_dir)
        self._try_load_models()
    
    def _build_feature_vector(
        self,
        query_complexity: Dict[str, int],
        graph_features: Dict[str, Any],
        resource: Dict[str, Any]
    ) -> np.ndarray:
        """Build feature vector from components."""
        query_feats = [
            query_complexity.get('v_scan', 0),
            query_complexity.get('e_scan', 0),
            query_complexity.get('f_scan', 0),
            query_complexity.get('atomic', 0),
            query_complexity.get('sync', 0),
        ]
        
        graph_feats = [
            graph_features.get('num_vertices', 0),
            graph_features.get('num_edges', 0),
            graph_features.get('avg_degree', 0),
            graph_features.get('max_degree', 0),
            graph_features.get('density', 0),
            graph_features.get('avg_clustering', 0),
        ]
        
        resource_feats = [
            resource.get('cpu_cores', 0),
            resource.get('memory_gb', 0),
            resource.get('num_gpus', 0),
            resource.get('gpu_memory_gb', 0),
            resource.get('storage_gb', 0),
        ]
        
        return np.array(query_feats + graph_feats + resource_feats)
    
    def predict_time(
        self,
        query_complexity: Dict[str, int],
        graph_features: Dict[str, Any],
        resource: Dict[str, Any],
        return_uncertainty: bool = False
    ) -> Tuple[float, ...]:
        """
        Predict execution time.
        
        Args:
            query_complexity: Query complexity dictionary
            graph_features: Graph features dictionary
            resource: Resource configuration dictionary
            return_uncertainty: Whether to return confidence interval
            
        Returns:
            If return_uncertainty: (time, lower, upper)
            Else: (time,)
        """
        if self.time_model is None:
            raise ValueError("Time model not loaded. Train or load models first.")
        
        X = self._build_feature_vector(query_complexity, graph_features, resource)
        X = X.reshape(1, -1)
        
        if return_uncertainty:
            pred, lower, upper = self.time_model.predict_with_uncertainty(X)
            return float(pred[0]), float(lower[0]), float(upper[0])
        else:
            pred = self.time_model.predict(X)
            return float(pred[0])
    
    def predict_cost(
        self,
        query_complexity: Dict[str, int],
        graph_features: Dict[str, Any],
        resource: Dict[str, Any],
        return_uncertainty: bool = False
    ) -> Tuple[float, ...]:
        """
        Predict execution cost.
        
        Args:
            query_complexity: Query complexity dictionary
            graph_features: Graph features dictionary
            resource: Resource configuration dictionary
            return_uncertainty: Whether to return confidence interval
            
        Returns:
            If return_uncertainty: (cost, lower, upper)
            Else: (cost,)
        """
        if self.cost_model is None:
            raise ValueError("Cost model not loaded. Train or load models first.")
        
        X = self._build_feature_vector(query_complexity, graph_features, resource)
        X = X.reshape(1, -1)
        
        if return_uncertainty:
            pred, lower, upper = self.cost_model.predict_with_uncertainty(X)
            return float(pred[0]), float(lower[0]), float(upper[0])
        else:
            pred = self.cost_model.predict(X)
            return float(pred[0])
    
    def predict_both(
        self,
        query_complexity: Dict[str, int],
        graph_features: Dict[str, Any],
        resource: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Predict both time and cost.

        Args:
            query_complexity: Query complexity dictionary
            graph_features: Graph features dictionary
            resource: Resource configuration dictionary

        Returns:
            Dictionary with time and cost predictions
        """
        result = {}

        if self.time_model is not None:
            t, t_low, t_up = self.predict_time(
                query_complexity, graph_features, resource, return_uncertainty=True
            )
            result['execution_time_ms'] = t
            result['execution_time_lower'] = t_low
            result['execution_time_upper'] = t_up
        else:
            # Heuristic fallback
            result.update(self._heuristic_time(query_complexity, graph_features, resource))

        if self.cost_model is not None:
            c, c_low, c_up = self.predict_cost(
                query_complexity, graph_features, resource, return_uncertainty=True
            )
            result['execution_cost'] = c
            result['execution_cost_lower'] = c_low
            result['execution_cost_upper'] = c_up
        else:
            # Heuristic fallback
            result.update(self._heuristic_cost(query_complexity, graph_features, resource))
        
        return result
    
    def _heuristic_time(
        self,
        query_complexity: Dict[str, int],
        graph_features: Dict[str, Any],
        resource: Dict[str, Any]
    ) -> Dict[str, float]:
        """Heuristic time prediction when model not available."""
        V = graph_features.get('num_vertices', 1000)
        E = graph_features.get('num_edges', 5000)
        cpu = resource.get('cpu_cores', 8)
        gpu = resource.get('num_gpus', 0)
        
        # Query complexity factor
        q_factor = sum(query_complexity.values()) or 1
        
        base_time = (V * 0.01 + E * 0.001) * (32 / max(cpu, 1)) * q_factor
        if gpu > 0:
            base_time *= 0.5 / resource['num_gpus']
        
        return {
            'execution_time_ms': base_time,
            'execution_time_lower': base_time * 0.8,
            'execution_time_upper': base_time * 1.2,
        }
    
    def _heuristic_cost(
        self,
        query_complexity: Dict[str, int],
        graph_features: Dict[str, Any],
        resource: Dict[str, Any]
    ) -> Dict[str, float]:
        """Heuristic cost prediction when model not available."""
        V = graph_features.get('num_vertices', 1000)
        E = graph_features.get('num_edges', 5000)
        cpu = resource.get('cpu_cores', 8)
        gpu = resource.get('num_gpus', 0)
        
        base_time = (V * 0.01 + E * 0.001) * (32 / max(cpu, 1))
        if gpu > 0:
            base_time *= 0.5
        
        cost = base_time * cpu * 0.05
        
        return {
            'execution_cost': cost,
            'execution_cost_lower': cost * 0.8,
            'execution_cost_upper': cost * 1.2,
        }

    def predict_batch(
        self,
        samples: List[Dict[str, Any]]
    ) -> List[Dict[str, float]]:
        """
        Batch prediction for multiple samples.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for sample in samples:
            pred = self.predict_both(
                sample['query']['complexity'],
                sample['graph_features'],
                sample['config']['resource']
            )
            results.append(pred)
        return results
