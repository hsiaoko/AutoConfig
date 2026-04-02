"""
AutoConfig - Graph Query Configuration System

This package provides tools for:
1. Feature Extraction: Extract static, symbolic, and graph features
2. Offline Training: Generate data and train Bayesian models
3. Online Recommendation: Predict cost and recommend optimal configurations

Subsystems:
- feature_extractor: Query, graph, and config feature extraction
- offline: Data generation, model training, registry
- online: Cost prediction, optimization, recommendation
- models: Bayesian time and cost models
"""

# Feature extraction
from .feature_extractor import (
    StaticFeatureExtractor,
    SymbolicFeatureExtractor,
    GraphPartitionExtractor,
    ConfigFeatureExtractor,
    FeatureManager,
)

# Models
from .models import BayesianExecutionTimeModel
from .models.bayesian_models import BayesianTimeModel, BayesianCostModel

# Prediction (legacy compatibility)
from .prediction import CostPredictor as LegacyCostPredictor

# Offline training
from .offline import DataGenerator, Trainer, ModelRegistry

# Online recommendation
from .online import (
    CostPredictor,
    ConfigurationOptimizer,
    Recommender,
)

__version__ = '2.0.0'
__author__ = 'AutoConfig Team'

__all__ = [
    # Feature extraction
    'StaticFeatureExtractor',
    'SymbolicFeatureExtractor',
    'GraphPartitionExtractor',
    'ConfigFeatureExtractor',
    'FeatureManager',
    
    # Models
    'BayesianExecutionTimeModel',
    'BayesianTimeModel',
    'BayesianCostModel',
    
    # Offline
    'DataGenerator',
    'Trainer',
    'ModelRegistry',
    
    # Online
    'CostPredictor',
    'ConfigurationOptimizer',
    'Recommender',
    
    # Legacy compatibility
    'LegacyCostPredictor',
]
