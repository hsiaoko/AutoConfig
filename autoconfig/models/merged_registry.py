"""
可插拔 **merged 表格式回归** 后端注册表。

* ``bayesian`` — :class:`BayesianCostModel`（默认）
* ``mlp`` / ``nn`` — 神经网络（`MLPRegressorBackend`：默认对 **X**、**y** 做 ``StandardScaler`` 再训练 sklearn ``MLPRegressor``，避免 merged 特征与 ``cost`` 量级差异过大导致损失爆炸；可用 ``model_options["scale_xy"]=false`` 关闭。两者等价，推荐 CLI 用 ``nn``）
* ``rl`` — 占位，训练会显式报错，便于以后接强化学习 / 自定义实现

使用 :func:`register_model_kind` 在运行时挂接自定义类。
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from .bayesian_models import BayesianCostModel
from .protocols import MergedTabularRegressor

# (factory, loader)  — loader(path:str)->MergedTabularRegressor
_HANDLERS: dict[str, tuple[Callable[..., Any], Callable[[str], Any]]] = {}


def _default_bayesian_factory(*, n_iter: int = 300, **kw: Any) -> BayesianCostModel:
    if kw:
        return BayesianCostModel(n_iter=n_iter, **kw)
    return BayesianCostModel(n_iter=n_iter)


def _default_bayesian_load(path: str) -> BayesianCostModel:
    return BayesianCostModel.load(path)


def _mlp_factory(**kw: Any):
    from .backends.mlp_regressor import MLPRegressorBackend

    return MLPRegressorBackend(**kw)


def _mlp_load(path: str):
    from .backends.mlp_regressor import MLPRegressorBackend

    return MLPRegressorBackend.load(path)


def _rl_factory(**kw: Any):
    from .backends.rl_placeholder import RLRegressorPlaceholder

    return RLRegressorPlaceholder(**kw)


def _rl_load(_path: str):
    raise NotImplementedError(
        "model_kind='rl' 为占位，无可序列化实现；请使用自定义 MergedTabularRegressor 并 register_model_kind。"
    )


def register_model_kind(
    kind: str,
    factory: Callable[..., MergedTabularRegressor],
    load: Callable[[str], MergedTabularRegressor],
) -> None:
    """
    注册或覆盖一种 ``model_kind``（小写名字）。

    ``factory`` 接收 ``**model_options``；``load`` 从 ``.pkl`` / joblib 文件恢复模型。
    """
    _HANDLERS[kind.lower()] = (factory, load)


def _init_builtin() -> None:
    if _HANDLERS:
        return
    register_model_kind("bayesian", _default_bayesian_factory, _default_bayesian_load)
    register_model_kind("mlp", _mlp_factory, _mlp_load)
    register_model_kind("nn", _mlp_factory, _mlp_load)  # 与 mlp 同实现，面向 CLI 的简短名
    register_model_kind("rl", _rl_factory, _rl_load)


def available_model_kinds() -> list[str]:
    _init_builtin()
    return sorted(_HANDLERS.keys())


def create_model(kind: str, /, **options: Any) -> MergedTabularRegressor:
    """``kind`` 如 ``"bayesian"``；``**options`` 传入对应 factory（如 ``n_iter``、``hidden_layer_sizes``）。"""
    _init_builtin()
    k = kind.lower().strip()
    if k not in _HANDLERS:
        raise ValueError(
            f"Unknown model_kind {kind!r}. Choose one of: {available_model_kinds()}"
        )
    factory, _ = _HANDLERS[k]
    return factory(**options)


def load_merged_regressor_file(path: str, meta: Optional[dict[str, Any]] = None) -> MergedTabularRegressor:
    """
    从训练写出的单文件加载模型。

    * 若 ``meta`` 含 ``model_kind``，按注册表 ``load`` 分流。
    * 若无 meta（或缺字段），先尝试 **贝叶斯** pickle；失败则尝试 **joblib**（``mlp``）。
    """
    _init_builtin()
    from pathlib import Path

    p = Path(path)

    kind: Optional[str] = None
    if meta and meta.get("model_kind") is not None:
        kind = str(meta["model_kind"]).lower().strip()

    if kind and kind in _HANDLERS:
        _, loader = _HANDLERS[kind]
        return loader(str(p))

    # 启发式（旧 checkpoint 无 model_kind）
    try:
        return _HANDLERS["bayesian"][1](str(p))
    except Exception:
        pass
    try:
        import joblib

        d = joblib.load(str(p))
        if isinstance(d, dict) and d.get("model_kind") == "mlp":
            return _HANDLERS["mlp"][1](str(p))
    except Exception:
        pass
    return _HANDLERS["bayesian"][1](str(p))
