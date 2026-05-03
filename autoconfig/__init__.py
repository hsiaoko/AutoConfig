"""
AutoConfig — graph query features, config merge, and merged 53-D Bayesian training.

**Tools (by stage)**

- **Feature extraction:** :mod:`autoconfig.feature_extractor`, :mod:`autoconfig.utils` extractors
- **Merge:** :class:`autoconfig.utils.feature_merger.FeatureMerger`
- **Merged training / eval / inference:** :mod:`autoconfig.merged` (``train-merged`` / ``eval-merged`` CLI;
  :class:`MergedBayesianPredictor` for ``bayesian_cost_merged.pkl``). **Config search:** :func:`recommend_top_k` in :mod:`autoconfig.conf_recommend`.

Config YAML generation (e.g. LHS) lives under ``data/conf/`` as standalone scripts, not in this import graph.
"""

from .feature_extractor import (
    ConfigFeatureExtractor,
    FeatureManager,
    GraphPartitionExtractor,
    StaticFeatureExtractor,
    SymbolicFeatureExtractor,
)
from .conf_recommend import (
    ConfRecommendation,
    ConfRecommendResult,
    export_recommendations_to_dir,
    perturb_configuration,
    recommend_top_k,
)
from .merged import MergedBayesianPredictor, load_bayesian_cost_merged
from .models import BayesianExecutionTimeModel
from .models.bayesian_models import BayesianCostModel, BayesianTimeModel

__version__ = "2.0.0"
__author__ = "AutoConfig Team"

__all__ = [
    "ConfRecommendation",
    "ConfRecommendResult",
    "export_recommendations_to_dir",
    "perturb_configuration",
    "recommend_top_k",
    "StaticFeatureExtractor",
    "SymbolicFeatureExtractor",
    "GraphPartitionExtractor",
    "ConfigFeatureExtractor",
    "FeatureManager",
    "BayesianExecutionTimeModel",
    "BayesianTimeModel",
    "BayesianCostModel",
    "MergedBayesianPredictor",
    "load_bayesian_cost_merged",
]
