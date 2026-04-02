"""
Trainer Module

Trains Bayesian models for execution time and cost prediction.
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from .data_generator import DataGenerator
from ..feature_extractor import FeatureManager
from ..models.bayesian_models import BayesianTimeModel, BayesianCostModel


class Trainer:
    """
    Train Bayesian models for time and cost prediction.
    
    Pipeline:
    1. Load or generate training data
    2. Extract features from query/graph/config
    3. Train time prediction model
    4. Train cost prediction model
    5. Save models to registry
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize trainer.
        
        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = Path(model_dir) if model_dir else Path('data/models')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.feature_manager = FeatureManager()
        self.data_generator = DataGenerator()
        
        self.time_model = BayesianTimeModel()
        self.cost_model = BayesianCostModel()
        
        self.training_metrics = {}
    
    def prepare_features(
        self,
        samples: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features from samples.
        
        Args:
            samples: List of sample dictionaries from DataGenerator
            
        Returns:
            Tuple of (feature_matrix, time_labels, cost_labels)
        """
        features_list = []
        time_labels = []
        cost_labels = []
        
        for sample in samples:
            # Extract features
            query_code = sample['query']['code']
            graph_features = sample['graph_features']
            config = sample['config']
            
            # Build feature vector
            feature_vec = self._build_feature_vector(
                sample['query']['complexity'],
                graph_features,
                config['resource']
            )
            features_list.append(feature_vec)
            
            # Collect labels
            time_labels.append(sample['labels']['execution_time_ms'])
            cost_labels.append(sample['labels']['execution_cost'])
        
        return (
            np.array(features_list),
            np.array(time_labels),
            np.array(cost_labels)
        )
    
    def _build_feature_vector(
        self,
        query_complexity: Dict[str, int],
        graph_features: Dict[str, Any],
        resource: Dict[str, Any]
    ) -> np.ndarray:
        """
        Build a feature vector from query, graph, and config.
        
        Features:
        - Query complexity (5 features)
        - Graph statistics (6 features)
        - Resource configuration (5 features)
        """
        # Query complexity features
        query_feats = [
            query_complexity.get('v_scan', 0),
            query_complexity.get('e_scan', 0),
            query_complexity.get('f_scan', 0),
            query_complexity.get('atomic', 0),
            query_complexity.get('sync', 0),
        ]
        
        # Graph features
        graph_feats = [
            graph_features.get('num_vertices', 0),
            graph_features.get('num_edges', 0),
            graph_features.get('avg_degree', 0),
            graph_features.get('max_degree', 0),
            graph_features.get('density', 0),
            graph_features.get('avg_clustering', 0),
        ]
        
        # Resource features
        resource_feats = [
            resource.get('cpu_cores', 0),
            resource.get('memory_gb', 0),
            resource.get('num_gpus', 0),
            resource.get('gpu_memory_gb', 0),
            resource.get('storage_gb', 0),
        ]
        
        return np.array(query_feats + graph_feats + resource_feats)
    
    def train(
        self,
        dataset_path: Optional[str] = None,
        num_samples: int = 500,
        test_split: float = 0.2,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train both time and cost models.
        
        Args:
            dataset_path: Path to existing dataset or None to generate
            num_samples: Number of samples if generating
            test_split: Fraction of data for testing
            verbose: Print progress
            
        Returns:
            Training metrics dictionary
        """
        if verbose:
            print("=" * 60)
            print("Training Bayesian Models")
            print("=" * 60)
        
        # Load or generate data
        if dataset_path and Path(dataset_path).exists():
            if verbose:
                print(f"\nLoading dataset from {dataset_path}")
            samples, _ = DataGenerator.load_dataset(dataset_path)
        else:
            if verbose:
                print(f"\nGenerating {num_samples} synthetic samples...")
            samples = self.data_generator.generate_dataset(num_samples)
        
        # Prepare features
        if verbose:
            print("Extracting features...")
        X, y_time, y_cost = self.prepare_features(samples)
        
        # Split data
        n_test = int(len(X) * test_split)
        indices = np.random.permutation(len(X))
        test_idx, train_idx = indices[:n_test], indices[n_test:]
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_time_train, y_time_test = y_time[train_idx], y_time[test_idx]
        y_cost_train, y_cost_test = y_cost[train_idx], y_cost[test_idx]
        
        if verbose:
            print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
            print(f"Feature dimension: {X.shape[1]}")
        
        # Train time model
        if verbose:
            print("\n[1/2] Training execution time model...")
        self.time_model.fit(X_train, y_time_train, verbose=verbose)
        time_train_r2 = self.time_model.score(X_train, y_time_train)
        time_test_r2 = self.time_model.score(X_test, y_time_test)
        
        if verbose:
            print(f"  Train R²: {time_train_r2:.4f}")
            print(f"  Test R²: {time_test_r2:.4f}")
        
        # Train cost model
        if verbose:
            print("\n[2/2] Training execution cost model...")
        self.cost_model.fit(X_train, y_cost_train, verbose=verbose)
        cost_train_r2 = self.cost_model.score(X_train, y_cost_train)
        cost_test_r2 = self.cost_model.score(X_test, y_cost_test)
        
        if verbose:
            print(f"  Train R²: {cost_train_r2:.4f}")
            print(f"  Test R²: {cost_test_r2:.4f}")
        
        # Store metrics
        self.training_metrics = {
            'time_model': {
                'train_r2': float(time_train_r2),
                'test_r2': float(time_test_r2),
            },
            'cost_model': {
                'train_r2': float(cost_train_r2),
                'test_r2': float(cost_test_r2),
            },
            'dataset': {
                'total_samples': len(samples),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_dim': X.shape[1],
            },
            'trained_at': datetime.now().isoformat(),
        }
        
        # Save models
        self.save_models()
        
        if verbose:
            print("\n" + "=" * 60)
            print("Training Complete!")
            print(f"Models saved to: {self.model_dir}")
            print("=" * 60)
        
        return self.training_metrics
    
    def save_models(self):
        """Save trained models."""
        self.time_model.save(str(self.model_dir / 'time_model.pkl'))
        self.cost_model.save(str(self.model_dir / 'cost_model.pkl'))
        
        # Save training metrics
        metrics_file = self.model_dir / 'training_metrics.yaml'
        with open(metrics_file, 'w') as f:
            yaml.dump(self.training_metrics, f, default_flow_style=None)
    
    def load_models(self, model_dir: Optional[str] = None):
        """Load trained models."""
        model_dir = Path(model_dir) if model_dir else self.model_dir
        
        time_path = model_dir / 'time_model.pkl'
        cost_path = model_dir / 'cost_model.pkl'
        
        if not time_path.exists() or not cost_path.exists():
            raise FileNotFoundError(f"Models not found in {model_dir}")
        
        self.time_model = BayesianTimeModel.load(str(time_path))
        self.cost_model = BayesianCostModel.load(str(cost_path))
        
        # Load metrics if available
        metrics_file = model_dir / 'training_metrics.yaml'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                self.training_metrics = yaml.safe_load(f)
