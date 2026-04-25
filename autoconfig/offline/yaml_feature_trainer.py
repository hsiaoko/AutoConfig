"""
Train BayesianCostModel on merged feature YAMLs (e.g. from ``merge`` / ``out/train``).

Y: first feature (``cost``). X: remaining ``num_features - 1`` columns.
Reuses :class:`autoconfig.models.bayesian_models.BayesianCostModel`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from ..models.bayesian_models import BayesianCostModel

_DEFAULT_COST_NAME = "cost"


def _row_keep_mask(
    feature_names_x: List[str], *, exclude_static: bool, exclude_symbolic: bool
) -> np.ndarray:
    """
    Merged X columns follow FeatureMerger order: static_*, sym_*, graph_*/partition_*, conf_*.
    `exclude_static` drops static (SPF) program features; `exclude_symbolic` drops
    graph-parameterized symbolic (SGF) features.
    """
    keep: List[bool] = []
    for name in feature_names_x:
        if name.startswith("static_"):
            keep.append(not exclude_static)
        elif name.startswith("sym_"):
            keep.append(not exclude_symbolic)
        else:
            keep.append(True)
    mask = np.array(keep, dtype=bool)
    if not np.any(mask):
        raise ValueError(
            "Exclusions would remove all inputs; at least graph or config features are required"
        )
    return mask


def apply_merged_feature_exclusions(
    table: MergedFeatureTable,
    *,
    exclude_static: bool = False,
    exclude_symbolic: bool = False,
) -> MergedFeatureTable:
    """
    Return a new table with columns removed according to name prefixes; ``y`` unchanged.
    """
    if not exclude_static and not exclude_symbolic:
        return table
    mask = _row_keep_mask(
        table.feature_names_x,
        exclude_static=exclude_static,
        exclude_symbolic=exclude_symbolic,
    )
    names_f = [n for n, k in zip(table.feature_names_x, mask) if k]
    X_f = table.X[:, mask]
    return MergedFeatureTable(
        X=np.ascontiguousarray(X_f, dtype=np.float64),
        y=table.y,
        feature_names_x=names_f,
        target_name=table.target_name,
        source_files=table.source_files,
    )


@dataclass(frozen=True)
class MergedFeatureTable:
    """Tabular data parsed from one or more merged feature YAML files."""

    X: np.ndarray
    y: np.ndarray
    feature_names_x: List[str]
    target_name: str
    source_files: List[str]


def _as_floats(seq: Sequence) -> List[float]:
    return [float(x) for x in seq]


def load_merged_feature_dir(
    data_dir: str | Path,
    *,
    pattern: str = "*.yaml",
    target_name: str = _DEFAULT_COST_NAME,
) -> MergedFeatureTable:
    """
    Load all ``pattern`` files under ``data_dir``. Each file must have matching
    ``feature_names`` and ``feature_vector``; first feature is the regression target
    (default name ``cost``), the rest are inputs.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    paths = sorted(root.glob(pattern), key=lambda p: p.name)
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} under {root}")

    ref_names: Optional[List[str]] = None
    X_rows: List[List[float]] = []
    y_col: List[float] = []
    sources: List[str] = []

    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        if not isinstance(doc, dict):
            raise ValueError(f"Expected mapping at root of {p}")

        names = doc.get("feature_names")
        vec = doc.get("feature_vector")
        nfeat = doc.get("num_features")

        if not isinstance(names, list) or not isinstance(vec, list):
            raise ValueError(f"Missing feature_names or feature_vector in {p}")
        if nfeat is not None and (int(nfeat) != len(names) or int(nfeat) != len(vec)):
            raise ValueError(
                f"num_features mismatch with names/vector lengths in {p}"
            )
        if len(names) < 2:
            raise ValueError(f"Need at least 2 features (target + 1 input) in {p}")
        if len(names) != len(vec):
            raise ValueError(f"feature_names and feature_vector length differ in {p}")

        if ref_names is None:
            ref_names = [str(x) for x in names]
            if str(ref_names[0]) != target_name:
                raise ValueError(
                    f"First feature in {p} is {ref_names[0]!r}, expected {target_name!r}"
                )
        else:
            if [str(x) for x in names] != ref_names:
                raise ValueError(
                    f"feature_names differ from other files: {p} vs first file"
                )

        v = _as_floats(vec)
        y_col.append(v[0])
        X_rows.append(v[1:])

    assert ref_names is not None
    feature_names_x = [str(n) for n in ref_names[1:]]

    return MergedFeatureTable(
        X=np.array(X_rows, dtype=np.float64),
        y=np.array(y_col, dtype=np.float64),
        feature_names_x=feature_names_x,
        target_name=target_name,
        source_files=[str(p) for p in paths],
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(
        np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100.0
    )
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def train_bayesian_cost_from_merged_yamls(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    test_split: float = 0.2,
    seed: int = 42,
    pattern: str = "*.yaml",
    model_basename: str = "bayesian_cost_merged",
    exclude_static: bool = False,
    exclude_symbolic: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Fit :class:`BayesianCostModel` on merged YAMLs; save pickle + a small sidecar
    ``{model_basename}_meta.yaml`` (feature list, paths, metrics).
    All paths are taken from the caller; nothing is hardcoded to a project root.

    If ``exclude_static`` is True, drop ``static_*`` (static program) features. If
    ``exclude_symbolic`` is True, drop ``sym_*`` (graph-parameterized symbolic) features.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = load_merged_feature_dir(data_dir, pattern=pattern)
    table = apply_merged_feature_exclusions(
        table,
        exclude_static=exclude_static,
        exclude_symbolic=exclude_symbolic,
    )
    n = len(table.y)
    if n < 2 and test_split > 0:
        raise ValueError("Need at least 2 samples when test_split > 0")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(np.floor(n * test_split)) if test_split > 0 else 0
    test_idx = set(perm[:n_test].tolist()) if n_test else set()
    train_mask = np.array([i not in test_idx for i in range(n)], dtype=bool)

    X_train, y_train = table.X[train_mask], table.y[train_mask]
    X_test, y_test = table.X[~train_mask], table.y[~train_mask]

    model = BayesianCostModel()
    model.fit(X_train, y_train, verbose=verbose)

    y_hat_train = model.predict(X_train)
    train_metrics = _metrics(y_train, y_hat_train)

    out: dict[str, Any] = {
        "data_dir": str(data_dir),
        "pattern": pattern,
        "source_files": table.source_files,
        "n_total": n,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features_x": int(table.X.shape[1]),
        "exclude_static": bool(exclude_static),
        "exclude_symbolic": bool(exclude_symbolic),
        "target": table.target_name,
        "feature_names_x": table.feature_names_x,
        "train_metrics": train_metrics,
        "test_metrics": None,
    }

    if X_test.shape[0] > 0:
        y_hat_test = model.predict(X_test)
        out["test_metrics"] = _metrics(y_test, y_hat_test)
        if verbose:
            tm = out["test_metrics"]
            print(f"Test  MAE: {tm['mae']:.4f}  RMSE: {tm['rmse']:.4f}  R²: {tm['r2']:.4f}")
    else:
        if verbose:
            print("No test set (all samples used for training).")

    if verbose:
        tr = out["train_metrics"]
        print(f"Train MAE: {tr['mae']:.4f}  RMSE: {tr['rmse']:.4f}  R²: {tr['r2']:.4f}")

    pkl_name = f"{model_basename}.pkl"
    meta_name = f"{model_basename}_meta.yaml"
    model.save(str(output_dir / pkl_name))
    with open(output_dir / meta_name, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)

    if verbose:
        print(f"Wrote {output_dir / pkl_name}")
        print(f"Wrote {output_dir / meta_name}")

    return out


def load_bayesian_cost_merged(
    model_path: str | Path,
) -> Tuple[BayesianCostModel, dict[str, Any]]:
    """Load model saved by :func:`train_bayesian_cost_from_merged_yamls` and its meta."""
    model_path = Path(model_path)
    model = BayesianCostModel.load(str(model_path))
    meta_path = model_path.parent / f"{model_path.stem}_meta.yaml"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Expected meta YAML next to model: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    return model, meta


def _load_bayesian_cost_merged_model_only(model_path: str | Path) -> BayesianCostModel:
    return BayesianCostModel.load(str(model_path))


def _optional_meta_for_model(model_path: Path) -> Optional[dict[str, Any]]:
    meta_path = model_path.parent / f"{model_path.stem}_meta.yaml"
    if not meta_path.is_file():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_bayesian_cost_on_merged_dir(
    model_path: str | Path,
    data_dir: str | Path,
    *,
    pattern: str = "*.yaml",
    check_feature_names: bool = True,
) -> dict[str, Any]:
    """
    Load a :class:`BayesianCostModel` and merged YAMLs under ``data_dir`` (same layout
    as training). Return aggregate metrics and per-file predictions vs ``cost`` labels.

    If ``<stem>_meta.yaml`` exists next to the ``.pkl``, input feature names are
    compared when ``check_feature_names`` is True. Otherwise only the feature
    dimension must match the trained weights.
    """
    model_path = Path(model_path)
    data_dir = Path(data_dir)

    model = _load_bayesian_cost_merged_model_only(model_path)
    if not model.is_fitted or model.weights is None:
        raise ValueError(f"Model is not fitted: {model_path}")

    meta = _optional_meta_for_model(model_path)
    table = load_merged_feature_dir(data_dir, pattern=pattern)
    ex_s = bool(meta.get("exclude_static", False)) if meta else False
    ex_y = bool(meta.get("exclude_symbolic", False)) if meta else False
    table = apply_merged_feature_exclusions(
        table, exclude_static=ex_s, exclude_symbolic=ex_y
    )
    n_in = int(table.X.shape[1])
    n_w = int(len(model.weights))
    if n_in != n_w:
        raise ValueError(
            f"Input feature count {n_in} != model weight length {n_w} "
            f"(check test data layout matches training)"
        )
    if (
        check_feature_names
        and meta
        and meta.get("feature_names_x") is not None
    ):
        if list(meta["feature_names_x"]) != list(table.feature_names_x):
            raise ValueError(
                "feature_names_x in test set differ from model meta. "
                f"Set check_feature_names=False to skip (not recommended)."
            )

    y_pred = model.predict(table.X)
    y_true = table.y
    m = _metrics(y_true, y_pred)

    per_file: list[dict[str, Any]] = []
    for i, path in enumerate(table.source_files):
        per_file.append(
            {
                "file": path,
                "y_true": float(y_true[i]),
                "y_pred": float(y_pred[i]),
                "abs_error": float(abs(y_true[i] - y_pred[i])),
            }
        )

    return {
        "model_path": str(model_path),
        "data_dir": str(data_dir),
        "pattern": pattern,
        "n_samples": int(len(y_true)),
        "metrics": m,
        "per_file": per_file,
    }
