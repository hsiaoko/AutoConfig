"""
Optional **code-path** cost prediction: ``FeatureManager`` + ``BayesianExecutionTimeModel``.

For **merged 53-D YAML** models (``bayesian_cost_merged.pkl``), use :mod:`autoconfig.merged` instead.
"""

from .cost_predictor import CostPredictor

__all__ = ["CostPredictor"]
