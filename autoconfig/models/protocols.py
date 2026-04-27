"""
Protocol for **tabular** merged-feature regressors (``train-merged`` / ``eval-merged``).

Any backend (贝叶斯、MLP、自研 RL 策略包装等) 实现相同接口即可接入 :mod:`autoconfig.merged.pipeline`。
"""

from __future__ import annotations

from typing import Any, Protocol, Tuple, Union, runtime_checkable

import numpy as np


@runtime_checkable
class MergedTabularRegressor(Protocol):
    """Fitted on ``(X, y)`` with shape ``(n, d)`` / ``(n,)``; saves next to ``*_meta.yaml``."""

    is_fitted: bool

    @property
    def n_input_features(self) -> int:
        """Input dimension ``d``; 0 if not yet fitted (where applicable)."""
        ...

    def fit(
        self, X: np.ndarray, y: np.ndarray, verbose: bool = False
    ) -> Any: ...

    def predict(
        self, X: np.ndarray, return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: ...

    def save(self, filepath: str) -> None: ...


# Optional: for static typing, concrete classes use @classmethod load
