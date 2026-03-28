"""
Cost Predictor
Predicts execution time (cost) for given Q, G, and Config.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import networkx as nx

from ..feature_extractor import FeatureManager
from ..models import BayesianExecutionTimeModel


class CostPredictor:
    """
    Main predictor class for graph query execution time.
    
    Combines feature extraction and Bayesian model for prediction.
    Supports:
    - Single and batch predictions
    - Uncertainty estimation
    - Model training and evaluation
    """
    
    def __init__(self, model: BayesianExecutionTimeModel = None):
        """
        Initialize cost predictor.
        
        Args:
            model: Pre-trained Bayesian model (optional)
        """
        self.feature_manager = FeatureManager()
        self.model = model or BayesianExecutionTimeModel()
    
    def predict(
        self,
        query: Union[str, nx.Graph, Dict[str, Any]],
        graph: Union[nx.Graph, Dict[str, Any]],
        config: Dict[str, Any],
        return_uncertainty: bool = False
    ) -> Union[float, Tuple[float, float, float]]:
        """
        Predict execution time for a single query-graph-config combination.

        Args:
            query: Query source code string, or Query graph (NetworkX Graph or dict)
            graph: Data graph (NetworkX Graph or dict with 'nodes' and 'edges')
            config: Configuration dictionary
            return_uncertainty: Whether to return confidence intervals

        Returns:
            Predicted execution time or (prediction, lower_bound, upper_bound)
        """
        # Extract features
        if isinstance(query, str):
            # Source code query
            features = self.feature_manager.extract_all(query, graph, config)
        elif isinstance(query, nx.Graph):
            features = self.feature_manager.extract_all(query, graph, config)
        else:
            features = self.feature_manager.extract_all_from_dict(query, graph, config)

        # Reshape for single sample
        features = features.reshape(1, -1)

        # Predict
        if return_uncertainty:
            pred, lower, upper = self.model.predict_with_uncertainty(features)
            return float(pred[0]), float(lower[0]), float(upper[0])
        else:
            pred = self.model.predict(features)
            return float(pred[0])
    
    def predict_batch(
        self,
        queries: List[Union[str, nx.Graph, Dict[str, Any]]],
        graphs: List[Union[nx.Graph, Dict[str, Any]]],
        configs: List[Dict[str, Any]],
        return_uncertainty: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Predict execution times for multiple query-graph-config combinations.

        Args:
            queries: List of query source code strings or query graphs
            graphs: List of data graphs
            configs: List of configurations
            return_uncertainty: Whether to return confidence intervals

        Returns:
            Array of predictions or (predictions, lower_bounds, upper_bounds)
        """
        n_samples = len(queries)

        # Extract all features
        features_list = []
        for i in range(n_samples):
            if isinstance(queries[i], str):
                # Source code query
                features = self.feature_manager.extract_all(
                    queries[i], graphs[i], configs[i]
                )
            elif isinstance(queries[i], nx.Graph):
                features = self.feature_manager.extract_all(
                    queries[i], graphs[i], configs[i]
                )
            else:
                features = self.feature_manager.extract_all_from_dict(
                    queries[i], graphs[i], configs[i]
                )
            features_list.append(features)

        features_matrix = np.vstack(features_list)

        # Predict
        if return_uncertainty:
            pred, lower, upper = self.model.predict_with_uncertainty(features_matrix)
            return pred, lower, upper
        else:
            pred = self.model.predict(features_matrix)
            return pred
    
    def train(
        self,
        queries: List[Union[str, nx.Graph, Dict[str, Any]]],
        graphs: List[Union[nx.Graph, Dict[str, Any]]],
        configs: List[Dict[str, Any]],
        execution_times: np.ndarray,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Train the model on given data.

        Args:
            queries: List of query source code strings or query graphs
            graphs: List of data graphs
            configs: List of configurations
            execution_times: Array of actual execution times
            verbose: Whether to print training progress

        Returns:
            Training metrics dictionary
        """
        n_samples = len(queries)

        # Extract all features
        features_list = []
        for i in range(n_samples):
            if isinstance(queries[i], str):
                # Source code query
                features = self.feature_manager.extract_all(
                    queries[i], graphs[i], configs[i]
                )
            elif isinstance(queries[i], nx.Graph):
                features = self.feature_manager.extract_all(
                    queries[i], graphs[i], configs[i]
                )
            else:
                features = self.feature_manager.extract_all_from_dict(
                    queries[i], graphs[i], configs[i]
                )
            features_list.append(features)

        features_matrix = np.vstack(features_list)

        # Train model
        self.model.fit(features_matrix, execution_times, verbose=verbose)

        # Compute training metrics
        predictions = self.model.predict(features_matrix)
        metrics = self._compute_metrics(execution_times, predictions)
        
        if verbose:
            print(f"Training completed:")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAPE: {metrics['mape']:.4f}")
            print(f"  R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def evaluate(
        self,
        queries: List[Union[nx.Graph, Dict[str, Any]]],
        graphs: List[Union[nx.Graph, Dict[str, Any]]],
        configs: List[Dict[str, Any]],
        execution_times: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            queries: List of query graphs
            graphs: List of data graphs
            configs: List of configurations
            execution_times: Array of actual execution times
            
        Returns:
            Evaluation metrics dictionary
        """
        predictions, lower, upper = self.predict_batch(
            queries, graphs, configs, return_uncertainty=True
        )
        
        metrics = self._compute_metrics(execution_times, predictions)
        
        # Compute calibration of uncertainty
        in_interval = (execution_times >= lower) & (execution_times <= upper)
        metrics['calibration'] = np.mean(in_interval)
        
        return metrics
    
    def _compute_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        # Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Root Mean Square Error
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # R² Score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2)
        }
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance based on model weights.
        
        Returns:
            Array of importance scores (absolute weight values)
        """
        if not self.model.is_fitted:
            raise ValueError("Model must be fitted first")
        
        return np.abs(self.model.weights)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        return self.feature_manager.get_feature_names()
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        self.model = BayesianExecutionTimeModel.load(filepath)
