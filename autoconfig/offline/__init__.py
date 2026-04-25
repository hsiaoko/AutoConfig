"""
Offline Training Module

Provides:
- Data generator for synthetic training data
- Bayesian model training
- Model registry and management
"""

from .data_generator import DataGenerator
from .trainer import Trainer
from .model_registry import ModelRegistry
from .yaml_feature_trainer import (
    MergedFeatureTable,
    load_merged_feature_dir,
    apply_merged_feature_exclusions,
    train_bayesian_cost_from_merged_yamls,
    load_bayesian_cost_merged,
    evaluate_bayesian_cost_on_merged_dir,
)

__all__ = [
    "DataGenerator",
    "Trainer",
    "ModelRegistry",
    "MergedFeatureTable",
    "load_merged_feature_dir",
    "apply_merged_feature_exclusions",
    "train_bayesian_cost_from_merged_yamls",
    "load_bayesian_cost_merged",
    "evaluate_bayesian_cost_on_merged_dir",
]
