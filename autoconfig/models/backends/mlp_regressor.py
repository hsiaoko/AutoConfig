"""sklearn ``MLPRegressor`` 包装，用于 ``model_kind='mlp'`` 的 merged 训练。"""

from __future__ import annotations

from typing import Any, Tuple, Union

import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _inner_mlp(est: Any) -> MLPRegressor:
    if isinstance(est, TransformedTargetRegressor):
        pipe = getattr(est, "regressor_", None) or est.regressor
        return pipe.named_steps["mlp"]
    return est


class MLPRegressorBackend:
    """
    全连接回归网络；与 :class:`BayesianCostModel` 一样实现 ``fit`` / ``predict`` / ``save``。

    默认对 **X**、**y** 做 ``StandardScaler``（merged 特征量级差异极大且 ``cost`` 常很大，
    直接喂给 ``MLPRegressor`` 会导致损失数值爆炸、收敛困难）。可通过构造参数或
    ``model_options`` 里 ``"scale_xy": false`` 关闭以兼容旧行为。

    不确定度：``return_std=True`` 时第二项为全零（与贝叶斯不同）。
    """

    _SERIAL_VERSION = 2

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (128, 64),
        max_iter: int = 500,
        random_state: int | None = 42,
        early_stopping: bool = True,
        scale_xy: bool = True,
        **mlp_kwargs: Any,
    ) -> None:
        self.scale_xy = bool(scale_xy)
        mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=early_stopping,
            **mlp_kwargs,
        )
        if self.scale_xy:
            pipe = Pipeline(
                steps=[
                    ("scale_x", StandardScaler()),
                    ("mlp", mlp),
                ]
            )
            self._estimator = TransformedTargetRegressor(
                regressor=pipe,
                transformer=StandardScaler(),
            )
        else:
            self._estimator = mlp
        self.is_fitted = False

    @property
    def n_input_features(self) -> int:
        if not self.is_fitted:
            return 0
        inner = _inner_mlp(self._estimator)
        ni = getattr(inner, "n_features_in_", None)
        return int(ni) if ni is not None else 0

    def fit(
        self, X: np.ndarray, y: np.ndarray, verbose: bool = False
    ) -> MLPRegressorBackend:
        _inner_mlp(self._estimator).set_params(verbose=verbose)
        self._estimator.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
        self.is_fitted = True
        return self

    def predict(
        self, X: np.ndarray, return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        p = np.asarray(self._estimator.predict(X), dtype=np.float64)
        if return_std:
            return p, np.zeros_like(p, dtype=np.float64)
        return p

    def save(self, filepath: str) -> None:
        if self.scale_xy and isinstance(self._estimator, TransformedTargetRegressor):
            payload: dict[str, Any] = {
                "model_kind": "mlp",
                "version": self._SERIAL_VERSION,
                "estimator": self._estimator,
                "is_fitted": self.is_fitted,
                "scale_xy": True,
            }
        else:
            payload = {
                "model_kind": "mlp",
                "mlp": self._estimator,
                "is_fitted": self.is_fitted,
                "scale_xy": False,
            }
        joblib.dump(payload, filepath)

    @classmethod
    def load(cls, filepath: str) -> MLPRegressorBackend:
        d = joblib.load(filepath)
        o = object.__new__(cls)
        o.is_fitted = bool(d.get("is_fitted", True))
        o.scale_xy = bool(d.get("scale_xy", False))
        if d.get("version") == cls._SERIAL_VERSION and "estimator" in d:
            o._estimator = d["estimator"]
            o.scale_xy = True
        elif "mlp" in d:
            o._estimator = d["mlp"]
        else:
            raise ValueError(f"Unrecognized MLP pickle layout in {filepath!r}")
        return o
