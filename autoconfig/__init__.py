"""
AutoConfig - Graph Query Execution Time Prediction

This package provides tools for predicting graph query execution times
given a query pattern (Q), data graph (G), and system configuration (Conf).

Subsystems:
1. Feature Extraction: Extract static, symbolic, and graph features
2. Model Training: Train Bayesian model for prediction
3. Prediction: Predict execution time (cost) for new Q, G, Conf
"""

from .feature_extractor import (
    StaticFeatureExtractor,
    SymbolicFeatureExtractor,
    GraphPartitionExtractor,
    ConfigFeatureExtractor,
    FeatureManager,
)
from .models import BayesianExecutionTimeModel
from .prediction import CostPredictor

__version__ = '2.0.0'
__author__ = 'AutoConfig Team'

__all__ = [
    'StaticFeatureExtractor',
    'SymbolicFeatureExtractor',
    'GraphPartitionExtractor',
    'ConfigFeatureExtractor',
    'FeatureManager',
    'BayesianExecutionTimeModel',
    'CostPredictor',
]
