"""sklearn ``MLPRegressor`` 包装，用于 ``model_kind='mlp'`` 的 merged 训练。"""

from __future__ import annotations

from typing import Any, Tuple, Union

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor


class MLPRegressorBackend:
    """
    全连接回归网络；与 :class:`BayesianCostModel` 一样实现 ``fit`` / ``predict`` / ``save``。

    不确定度：``return_std=True`` 时第二项为全零（与贝叶斯不同）。
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (128, 64),
        max_iter: int = 500,
        random_state: int | None = 42,
        early_stopping: bool = True,
        **mlp_kwargs: Any,
    ) -> None:
        self._mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=early_stopping,
            **mlp_kwargs,
        )
        self.is_fitted = False

    @property
    def n_input_features(self) -> int:
        if not self.is_fitted or self._mlp.n_features_in_ is None:
            return 0
        return int(self._mlp.n_features_in_)

    def fit(
        self, X: np.ndarray, y: np.ndarray, verbose: bool = False
    ) -> MLPRegressorBackend:
        self._mlp.set_params(verbose=verbose)
        self._mlp.fit(X, y)
        self.is_fitted = True
        return self

    def predict(
        self, X: np.ndarray, return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        p = np.asarray(self._mlp.predict(X), dtype=np.float64)
        if return_std:
            return p, np.zeros_like(p, dtype=np.float64)
        return p

    def save(self, filepath: str) -> None:
        joblib.dump(
            {
                "model_kind": "mlp",
                "mlp": self._mlp,
                "is_fitted": self.is_fitted,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str) -> MLPRegressorBackend:
        d = joblib.load(filepath)
        o = object.__new__(cls)
        o._mlp = d["mlp"]
        o.is_fitted = bool(d.get("is_fitted", True))
        return o
