"""
Merged 53-D YAML pipeline: load training tables, fit a :class:`~autoconfig.models.protocols.MergedTabularRegressor`
(default Bayesian), evaluate on directories, and run inference via :class:`MergedBayesianPredictor`.

Used by ``autoconfig train-merged`` / ``eval-merged`` and by
``./scripts/run_train_merged*.sh``, ``./scripts/run_eval_merged*.sh``.
"""

from ..models.merged_registry import available_model_kinds, create_model, register_model_kind
from .pipeline import (
    CONF_BATCH_SIZE_NAME,
    GRAPH_NUM_EDGES_NAME,
    MERGED_X_START_INDEX,
    MERGED_Y_AXIS_NAMES,
    VE_ONLY_GRAPH_NAMES,
    MergedFeatureTable,
    apply_conf_batch_size_override,
    apply_merged_feature_exclusions,
    evaluate_bayesian_cost_on_merged_dir,
    load_bayesian_cost_merged,
    load_merged_feature_dir,
    meta_exclusion_kwargs,
    optional_meta_for_model,
    train_bayesian_cost_from_merged_yamls,
    x_matrix_from_merged_docs,
    x_row_from_merged_doc,
)
from .predictor import MergedBayesianPredictor

__all__ = [
    "available_model_kinds",
    "create_model",
    "register_model_kind",
    "CONF_BATCH_SIZE_NAME",
    "GRAPH_NUM_EDGES_NAME",
    "MERGED_X_START_INDEX",
    "MERGED_Y_AXIS_NAMES",
    "VE_ONLY_GRAPH_NAMES",
    "MergedBayesianPredictor",
    "MergedFeatureTable",
    "apply_conf_batch_size_override",
    "apply_merged_feature_exclusions",
    "evaluate_bayesian_cost_on_merged_dir",
    "load_bayesian_cost_merged",
    "load_merged_feature_dir",
    "meta_exclusion_kwargs",
    "optional_meta_for_model",
    "train_bayesian_cost_from_merged_yamls",
    "x_matrix_from_merged_docs",
    "x_row_from_merged_doc",
]
