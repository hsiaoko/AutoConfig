"""
Bayesian Models for Execution Time and Cost Prediction

Two separate models:
1. BayesianTimeModel - predicts execution time
2. BayesianCostModel - predicts execution cost
"""

import numpy as np
from scipy import stats
import pickle
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path


class BayesianTimeModel:
    """
    Bayesian Ridge Regression for execution time prediction.
    
    Uses log-transformed target for better handling of skewed distributions.
    """
    
    def __init__(
        self,
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
        n_iter: int = 300,
        tol: float = 1e-3,
    ):
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.n_iter = n_iter
        self.tol = tol
        
        # Learned parameters
        self.weights = None
        self.sigma = None
        self.alpha = None
        self.lambda_ = None
        self.feature_means = None
        self.feature_stds = None
        self.is_fitted = False
    
    def _log_transform(self, y: np.ndarray) -> np.ndarray:
        return np.log(y + 1e-8)
    
    def _inverse_log_transform(self, y: np.ndarray) -> np.ndarray:
        return np.exp(y)
    
    def _normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0)
            self.feature_stds[self.feature_stds == 0] = 1.0
        return (X - self.feature_means) / self.feature_stds
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> 'BayesianTimeModel':
        """Fit the model."""
        n_samples, n_features = X.shape
        y_transformed = self._log_transform(y)
        X_normalized = self._normalize_features(X, fit=True)
        
        # Initialize
        self.weights = np.zeros(n_features)
        self.sigma = np.eye(n_features)
        self.alpha = 1.0
        self.lambda_ = 1.0
        
        # Variational inference
        for iteration in range(self.n_iter):
            old_weights = self.weights.copy()
            
            try:
                self.sigma = np.linalg.inv(
                    self.lambda_ * np.eye(n_features) +
                    self.alpha * X_normalized.T @ X_normalized
                )
            except np.linalg.LinAlgError:
                self.sigma = np.linalg.inv(
                    self.lambda_ * np.eye(n_features) +
                    self.alpha * X_normalized.T @ X_normalized +
                    1e-6 * np.eye(n_features)
                )
            
            self.weights = self.alpha * self.sigma @ X_normalized.T @ y_transformed
            
            self.lambda_ = (n_features + 2 * self.lambda_1) / (
                np.trace(self.sigma) + self.weights.T @ self.weights + 2 * self.lambda_2
            )
            
            self.alpha = (n_samples + 2 * self.alpha_1) / (
                np.sum((y_transformed - X_normalized @ self.weights) ** 2) + 2 * self.alpha_2
            )
            
            if np.linalg.norm(self.weights - old_weights) < self.tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray:
        """Predict execution time."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        X_normalized = self._normalize_features(X)
        y_pred_log = X_normalized @ self.weights
        y_pred = self._inverse_log_transform(y_pred_log)
        
        if return_std:
            y_var_log = 1.0 / self.alpha + np.sum(
                X_normalized @ self.sigma * X_normalized, axis=1
            )
            y_std = y_pred * np.sqrt(np.exp(y_var_log) - 1)
            return y_pred, y_std
        
        return y_pred
    
    def predict_with_uncertainty(
        self, X: np.ndarray, confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with confidence intervals."""
        y_pred, y_std = self.predict(X, return_std=True)
        z = stats.norm.ppf((1 + confidence_level) / 2)
        lower = np.maximum(0, y_pred - z * y_std)
        upper = y_pred + z * y_std
        return y_pred, lower, upper
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def save(self, filepath: str):
        """Save model to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'hyperparams': {
                    'alpha_1': self.alpha_1, 'alpha_2': self.alpha_2,
                    'lambda_1': self.lambda_1, 'lambda_2': self.lambda_2,
                    'n_iter': self.n_iter, 'tol': self.tol,
                },
                'params': {
                    'weights': self.weights, 'sigma': self.sigma,
                    'alpha': self.alpha, 'lambda': self.lambda_,
                    'feature_means': self.feature_means,
                    'feature_stds': self.feature_stds,
                    'is_fitted': self.is_fitted,
                }
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'BayesianTimeModel':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        model = cls(**data['hyperparams'])
        model.weights = data['params']['weights']
        model.sigma = data['params']['sigma']
        model.alpha = data['params']['alpha']
        model.lambda_ = data['params']['lambda']
        model.feature_means = data['params']['feature_means']
        model.feature_stds = data['params']['feature_stds']
        model.is_fitted = data['params']['is_fitted']
        return model


class BayesianCostModel(BayesianTimeModel):
    """
    Bayesian Ridge Regression for execution cost prediction.
    
    Same architecture as time model but trained on cost labels.
    """
    pass
