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

__all__ = ['DataGenerator', 'Trainer', 'ModelRegistry']
