"""
Load ``bayesian_cost_merged.pkl`` + ``*_meta.yaml`` and run batch or single prediction.

Use when you already have **aligned** input rows (``X`` with the same columns as
training), or full merged-feature YAML **dicts** (``feature_names`` + ``feature_vector``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from ..models.protocols import MergedTabularRegressor
from .pipeline import (
    load_bayesian_cost_merged,
    x_matrix_from_merged_docs,
    x_row_from_merged_doc,
)


class MergedBayesianPredictor:
    """
    Runtime wrapper around a trained :class:`MergedTabularRegressor` and its training
    ``meta`` dict (default backend remains Bayesian; name is historical).

    - **X-only prediction:** :meth:`predict_batch` / :meth:`predict_one` with rows matching
      ``len(meta['feature_names_x'])`` and training order.
    - **From merged YAML content:** :meth:`predict_merged_doc` / :meth:`predict_merged_docs`
      (applies the same feature exclusions and optional ``conf_batch_size`` as train/eval).
    """

    def __init__(self, model: MergedTabularRegressor, meta: dict[str, Any]) -> None:
        self._model = model
        self._meta = dict(meta) if meta else {}

    @classmethod
    def load(cls, model_path: str | Path) -> MergedBayesianPredictor:
        """Load ``.pkl`` and the sibling ``<stem>_meta.yaml`` (required)."""
        model, meta = load_bayesian_cost_merged(model_path)
        if not model.is_fitted or int(model.n_input_features) == 0:
            raise ValueError(f"Model in {model_path!r} is not fitted")
        return cls(model, meta)

    @property
    def model(self) -> MergedTabularRegressor:
        return self._model

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta

    @property
    def feature_names_x(self) -> Optional[List[str]]:
        return self._meta.get("feature_names_x")

    def _check_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n = int(self._model.n_input_features)
        if n <= 0:
            raise ValueError("Model has no input dimension (not fitted)")
        if int(X.shape[1]) != n:
            raise ValueError(
                f"X has {X.shape[1]} features but model expects {n} (see meta['feature_names_x'])"
            )
        return X

    def predict_batch(
        self,
        X: np.ndarray,
        *,
        return_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict for **pre-aligned** input matrix (each row = one training layout).

        ``X`` shape ``(n_samples, n_features)``; ``n_features`` must match the trained model.
        """
        Xb = self._check_X(X)
        return self._model.predict(Xb, return_std=return_std)

    def predict_one(
        self,
        x: np.ndarray,
        *,
        return_std: bool = False,
    ) -> Union[float, Tuple[float, float]]:
        """
        Single row, same as :meth:`predict_batch` with ``X`` of shape ``(1, n)`` or ``(n,)``.
        """
        out = self.predict_batch(x, return_std=return_std)
        if return_std:
            p, s = out
            return float(p[0]), float(s[0])
        return float(out[0])

    def predict_merged_doc(
        self,
        doc: dict[str, Any],
        *,
        conf_batch_size: Optional[float] = None,
    ) -> float:
        """One prediction from a merged feature YAML **object** (in-memory dict)."""
        X = x_row_from_merged_doc(
            doc, self._meta, conf_batch_size=conf_batch_size
        )
        return self.predict_one(X)

    def predict_merged_docs(
        self,
        docs: List[dict[str, Any]],
        *,
        conf_batch_size: Optional[float] = None,
    ) -> np.ndarray:
        """Batch prediction for several merged YAML **dicts** (same layout)."""
        if not docs:
            return np.array([], dtype=np.float64)
        X = x_matrix_from_merged_docs(
            docs, self._meta, conf_batch_size=conf_batch_size
        )
        return self.predict_batch(X, return_std=False)
