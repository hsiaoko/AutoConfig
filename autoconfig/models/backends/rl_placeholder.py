"""
RL 插槽：默认 **未实现** 训练。真实 RL 需环境 + 策略，请自行实现
:class:`autoconfig.models.protocols.MergedTabularRegressor` 并注册到
``autoconfig.models.merged_registry``。
"""


from __future__ import annotations

from typing import Any, Tuple, Union

import numpy as np


class RLRegressorPlaceholder:
    """
    占位类：不执行表格回归。用于占住 ``model_kind='rl'`` 并在 ``fit`` 时给出明确提示。

    若你已有 “状态 = 特征、动作 = 配置” 的 RL 管线，可子类化并实现 ``fit``/``predict``/``save``/``load``。
    """

    is_fitted: bool = False

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    @property
    def n_input_features(self) -> int:
        return 0

    def fit(
        self, X: np.ndarray, y: np.ndarray, verbose: bool = False
    ) -> RLRegressorPlaceholder:
        raise NotImplementedError(
            "内置 model_kind='rl' 未实现：强化学习需要自定义环境与奖励。请实现 "
            "MergedTabularRegressor 并在 autoconfig.models.merged_registry.register_model_kind 注册，"
            "或使用 'bayesian' / 'mlp'。"
        )

    def predict(
        self, X: np.ndarray, return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError("RL 占位类不支持 predict。")

    def save(self, filepath: str) -> None:
        raise NotImplementedError("RL 占位类不支持 save。")
