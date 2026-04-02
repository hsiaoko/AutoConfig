"""
Recommender Service

High-level interface for configuration recommendation.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import yaml

from .cost_predictor import CostPredictor
from .optimizer import ConfigurationOptimizer
from ..offline.data_generator import DataGenerator
from ..offline.trainer import Trainer


class Recommender:
    """
    Configuration recommendation service.
    
    Provides:
    - Single query recommendation
    - Batch recommendation
    - What-if analysis
    - Model management
    """
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        auto_train: bool = False
    ):
        """
        Initialize recommender.
        
        Args:
            model_dir: Directory for trained models
            auto_train: Whether to auto-train if models not found
        """
        self.model_dir = Path(model_dir) if model_dir else Path('data/models')
        self.cost_predictor = CostPredictor(self.model_dir)
        self.optimizer = ConfigurationOptimizer(self.cost_predictor)
        self.data_generator = DataGenerator()
        
        # Auto-train if needed
        if auto_train and not self.models_loaded:
            self.train_models()
    
    @property
    def models_loaded(self) -> bool:
        """Check if models are loaded."""
        return (self.cost_predictor.time_model is not None and
                self.cost_predictor.cost_model is not None)
    
    def train_models(
        self,
        dataset_path: Optional[str] = None,
        num_samples: int = 500,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train or retrain models.
        
        Args:
            dataset_path: Path to existing dataset or None to generate
            num_samples: Number of samples if generating
            verbose: Print progress
            
        Returns:
            Training metrics
        """
        trainer = Trainer(str(self.model_dir))
        metrics = trainer.train(
            dataset_path=dataset_path,
            num_samples=num_samples,
            verbose=verbose
        )
        
        # Reload models
        self.cost_predictor.load_models(str(self.model_dir))
        
        return metrics
    
    def recommend(
        self,
        query_name: str,
        graph_features: Dict[str, Any],
        candidate_configs: Optional[List[Dict[str, Any]]] = None,
        top_n: int = 3
    ) -> Dict[str, Any]:
        """
        Recommend best configuration for a query.

        Args:
            query_name: Name of query type (e.g., 'bfs', 'pagerank')
            graph_features: Graph features dictionary
            candidate_configs: Optional list of candidate configurations
            top_n: Number of top recommendations to return

        Returns:
            Recommendation result with top configurations
        """
        # Get query complexity
        if query_name not in self.data_generator.query_templates:
            raise ValueError(f"Unknown query type: {query_name}")

        query_complexity = self.data_generator.query_templates[query_name]['complexity']

        return self.recommend_with_complexity(
            query_complexity,
            graph_features,
            candidate_configs,
            top_n
        )

    def recommend_with_complexity(
        self,
        query_complexity: Dict[str, int],
        graph_features: Dict[str, Any],
        candidate_configs: Optional[List[Dict[str, Any]]] = None,
        top_n: int = 3
    ) -> Dict[str, Any]:
        """
        Recommend best configuration using query complexity directly.

        Args:
            query_complexity: Query complexity dictionary (v_scan, e_scan, etc.)
            graph_features: Graph features dictionary
            candidate_configs: Optional list of candidate configurations
            top_n: Number of top recommendations to return

        Returns:
            Recommendation result with diverse top configurations
        """
        # Run optimization
        result = self.optimizer.optimize(
            query_complexity,
            graph_features,
            candidate_configs,
            return_all=True
        )

        # Format recommendations from diverse_top_k
        recommendations = []
        for item in result.get('diverse_top_k', [])[:top_n]:
            rec = {
                'rank': item.get('rank', len(recommendations) + 1),
                'config_id': item['config']['config_id'],
                'resource': item['config']['resource'],
                'predicted_time_ms': item['predicted_time'],
                'predicted_cost': item['predicted_cost'],
                'is_perturbed': item.get('is_perturbed', False),
            }
            recommendations.append(rec)

        return {
            'query_complexity': query_complexity,
            'graph_features': graph_features,
            'recommendations': recommendations,
            'best_config': recommendations[0] if recommendations else None,
        }
    
    def recommend_for_graph(
        self,
        query_name: str,
        graph: Any,
        candidate_configs: Optional[List[Dict[str, Any]]] = None,
        top_n: int = 3
    ) -> Dict[str, Any]:
        """
        Recommend configuration for a specific graph.
        
        Args:
            query_name: Name of query type
            graph: NetworkX graph or graph features dict
            candidate_configs: Optional candidate configurations
            top_n: Number of recommendations
            
        Returns:
            Recommendation result
        """
        # Extract graph features if needed
        if hasattr(graph, 'number_of_nodes'):
            # NetworkX graph
            degrees = [d for _, d in graph.degree()]
            graph_features = {
                'num_vertices': graph.number_of_nodes(),
                'num_edges': graph.number_of_edges(),
                'avg_degree': np.mean(degrees) if degrees else 0,
                'max_degree': max(degrees) if degrees else 0,
                'density': 0,  # Can compute if needed
                'avg_clustering': 0,
            }
        else:
            graph_features = graph
        
        return self.recommend(query_name, graph_features, candidate_configs, top_n)
    
    def what_if(
        self,
        query_name: str,
        graph_features: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        What-if analysis for a specific configuration.
        
        Args:
            query_name: Name of query type
            graph_features: Graph features dictionary
            config: Configuration to analyze
            
        Returns:
            Prediction results
        """
        query_complexity = self.data_generator.query_templates[query_name]['complexity']
        
        predictions = self.cost_predictor.predict_both(
            query_complexity, graph_features, config['resource']
        )
        
        return {
            'query_name': query_name,
            'graph_features': graph_features,
            'config': config,
            'predictions': predictions,
        }
    
    def compare_configs(
        self,
        query_name: str,
        graph_features: Dict[str, Any],
        configs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare multiple configurations.
        
        Args:
            query_name: Name of query type
            graph_features: Graph features dictionary
            configs: List of configurations to compare
            
        Returns:
            Comparison results
        """
        query_complexity = self.data_generator.query_templates[query_name]['complexity']
        
        comparisons = []
        for config in configs:
            predictions = self.cost_predictor.predict_both(
                query_complexity, graph_features, config['resource']
            )
            comparisons.append({
                'config_id': config.get('config_id', 'unknown'),
                'resource': config['resource'],
                **predictions,
            })
        
        # Sort by cost
        comparisons.sort(key=lambda x: x.get('execution_cost', float('inf')))
        
        for i, c in enumerate(comparisons):
            c['rank'] = i + 1
        
        return {
            'query_name': query_name,
            'comparisons': comparisons,
        }
    
    def save_recommendation(
        self,
        result: Dict[str, Any],
        output_path: str
    ):
        """Save recommendation result to file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(result, f, default_flow_style=None)
