"""
Bayesian Execution Time Model
Implements Bayesian Ridge Regression for execution time prediction.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from scipy import stats
import pickle


class BayesianExecutionTimeModel:
    """
    Bayesian model for predicting graph query execution time.
    
    Uses Bayesian Ridge Regression which provides:
    - Automatic regularization parameter tuning
    - Uncertainty estimation for predictions
    - Robustness to overfitting
    
    The model predicts: log(execution_time) = f(features)
    Using log transformation for better handling of skewed time distributions.
    """
    
    def __init__(
        self,
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
        n_iter: int = 300,
        tol: float = 1e-3,
        use_log_transform: bool = True
    ):
        """
        Initialize Bayesian model.
        
        Args:
            alpha_1: Hyperparameter for alpha prior (Gamma distribution)
            alpha_2: Hyperparameter for alpha prior (Gamma distribution)
            lambda_1: Hyperparameter for lambda prior (Gamma distribution)
            lambda_2: Hyperparameter for lambda prior (Gamma distribution)
            n_iter: Maximum number of iterations
            tol: Convergence tolerance
            use_log_transform: Whether to apply log transform to target
        """
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.n_iter = n_iter
        self.tol = tol
        self.use_log_transform = use_log_transform
        
        # Model parameters (to be learned)
        self.weights = None  # Weight vector
        self.sigma = None  # Covariance matrix
        self.alpha = None  # Precision of noise
        self.lambda_ = None  # Precision of weights
        
        # Normalization parameters
        self.feature_means = None
        self.feature_stds = None
        
        # Training history
        self.training_history = []
        self.is_fitted = False
    
    def _log_transform(self, y: np.ndarray) -> np.ndarray:
        """Apply log transform to target values."""
        return np.log(y + 1e-8)
    
    def _inverse_log_transform(self, y: np.ndarray) -> np.ndarray:
        """Inverse log transform."""
        return np.exp(y)
    
    def _normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features using z-score."""
        if fit:
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0)
            self.feature_stds[self.feature_stds == 0] = 1.0
        
        X_normalized = (X - self.feature_means) / self.feature_stds
        return X_normalized
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        verbose: bool = False
    ) -> 'BayesianExecutionTimeModel':
        """
        Fit the Bayesian model using variational inference.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (execution times) (n_samples,)
            verbose: Whether to print training progress
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        
        # Transform target if needed
        if self.use_log_transform:
            y_transformed = self._log_transform(y)
        else:
            y_transformed = y
        
        # Normalize features
        X_normalized = self._normalize_features(X, fit=True)
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.sigma = np.eye(n_features)
        self.alpha = 1.0
        self.lambda_ = 1.0
        
        # Variational inference
        for iteration in range(self.n_iter):
            # Store old parameters for convergence check
            old_weights = self.weights.copy()
            
            # Update posterior covariance and mean
            try:
                self.sigma = np.linalg.inv(
                    self.lambda_ * np.eye(n_features) + 
                    self.alpha * X_normalized.T @ X_normalized
                )
            except np.linalg.LinAlgError:
                # Add regularization if singular
                self.sigma = np.linalg.inv(
                    self.lambda_ * np.eye(n_features) + 
                    self.alpha * X_normalized.T @ X_normalized + 
                    1e-6 * np.eye(n_features)
                )
            
            self.weights = self.alpha * self.sigma @ X_normalized.T @ y_transformed
            
            # Update hyperparameters
            self.lambda_ = (n_features + 2 * self.lambda_1) / (
                np.trace(self.sigma) + self.weights.T @ self.weights + 2 * self.lambda_2
            )
            
            self.alpha = (n_samples + 2 * self.alpha_1) / (
                np.sum((y_transformed - X_normalized @ self.weights) ** 2) + 
                2 * self.alpha_2
            )
            
            # Check convergence
            weight_change = np.linalg.norm(self.weights - old_weights)
            self.training_history.append({
                'iteration': iteration,
                'weight_change': weight_change,
                'alpha': self.alpha,
                'lambda': self.lambda_
            })
            
            if verbose and iteration % 50 == 0:
                print(f"Iteration {iteration}: weight_change={weight_change:.6f}")
            
            if weight_change < self.tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
        
        self.is_fitted = True
        return self
    
    def predict(
        self, 
        X: np.ndarray,
        return_std: bool = False
    ) -> np.ndarray:
        """
        Predict execution times.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            return_std: Whether to return prediction uncertainty
            
        Returns:
            Predicted execution times (n_samples,) or 
            Tuple of (predictions, standard deviations)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Normalize features
        X_normalized = self._normalize_features(X)
        
        # Predict in log space
        y_pred_log = X_normalized @ self.weights
        
        # Transform back
        if self.use_log_transform:
            y_pred = self._inverse_log_transform(y_pred_log)
        else:
            y_pred = y_pred_log
        
        if return_std:
            # Compute predictive variance
            y_var_log = 1.0 / self.alpha + np.sum(
                X_normalized @ self.sigma * X_normalized, axis=1
            )
            
            if self.use_log_transform:
                # Delta method for log-normal distribution
                y_std = y_pred * np.sqrt(np.exp(y_var_log) - 1)
            else:
                y_std = np.sqrt(y_var_log)
            
            return y_pred, y_std
        
        return y_pred
    
    def predict_with_uncertainty(
        self, 
        X: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty intervals.
        
        Args:
            X: Feature matrix
            confidence_level: Confidence level for intervals
            
        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        y_pred, y_std = self.predict(X, return_std=True)
        
        # Compute confidence intervals
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        lower_bound = np.maximum(0, y_pred - z_score * y_std)
        upper_bound = y_pred + z_score * y_std
        
        return y_pred, lower_bound, upper_bound
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score.
        
        Args:
            X: Feature matrix
            y: True execution times
            
        Returns:
            R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'weights': self.weights,
            'sigma': self.sigma,
            'alpha': self.alpha,
            'lambda': self.lambda_,
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'is_fitted': self.is_fitted
        }
    
    def set_params(self, params: Dict[str, Any]):
        """Set model parameters."""
        self.weights = params.get('weights', self.weights)
        self.sigma = params.get('sigma', self.sigma)
        self.alpha = params.get('alpha', self.alpha)
        self.lambda_ = params.get('lambda', self.lambda_)
        self.feature_means = params.get('feature_means', self.feature_means)
        self.feature_stds = params.get('feature_stds', self.feature_stds)
        self.is_fitted = params.get('is_fitted', False)
    
    def save(self, filepath: str):
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'hyperparams': {
                    'alpha_1': self.alpha_1,
                    'alpha_2': self.alpha_2,
                    'lambda_1': self.lambda_1,
                    'lambda_2': self.lambda_2,
                    'n_iter': self.n_iter,
                    'tol': self.tol,
                    'use_log_transform': self.use_log_transform
                },
                'params': self.get_params(),
                'training_history': self.training_history
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'BayesianExecutionTimeModel':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(**data['hyperparams'])
        model.set_params(data['params'])
        model.training_history = data.get('training_history', [])
        return model
