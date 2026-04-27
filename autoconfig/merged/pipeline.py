"""
Merged 53-D YAML training and evaluation pipeline (``train-merged`` / ``eval-merged``).

Default backend is :class:`autoconfig.models.bayesian_models.BayesianCostModel`;
other kinds (``mlp``, custom) are selected via ``model_kind`` / :mod:`autoconfig.models.merged_registry`.

**53-D merged layout:** ``feature_names`` must begin with ``price``, ``time``, ``cost``
(fixed order). The regression label **Y** is *one* of these three, chosen by
``y_axis`` (0 / 1 / 2) or the equivalent ``target_name`` (``price`` / ``time`` /
``cost``). The input matrix **X** always uses **columns 4–end** in file order
(0-based indices ``3..``), i.e. static → symbolic → graph/partition → config features — **not**
the other two label columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from ..models.merged_registry import create_model, load_merged_regressor_file
from ..models.protocols import MergedTabularRegressor

# First three names in :class:`~autoconfig.utils.feature_merger.FeatureMerger` order.
MERGED_Y_AXIS_NAMES: tuple[str, str, str] = ("price", "time", "cost")
# 0-based index of first *input* column (1-based features 4..53 in a 53-D vector).
MERGED_X_START_INDEX = 3

# no_pf (exclude static + symbolic): optional graph sparsity — only |V| and |E| scalars.
VE_ONLY_GRAPH_NAMES: frozenset[str] = frozenset(
    ("graph_num_vertices", "graph_num_edges")
)

# no_all (with graph_e_only): among graph_/partition_, only keep edge count.
GRAPH_NUM_EDGES_NAME = "graph_num_edges"

# Merged X column for system I/O batch size (see ConfigFeatureExtractor / FeatureMerger).
CONF_BATCH_SIZE_NAME = "conf_batch_size"

# Default label for merged layout (``time`` is usually filled from benchmarks).
_DEFAULT_TARGET = "time"


def _resolve_merged_y(
    target_name: str, y_axis: Optional[int]
) -> tuple[str, int]:
    """
    Return ``(name, index)`` for Y among the first three merged columns.
    If ``y_axis`` is set (0=price, 1=time, 2=cost), it overrides ``target_name``.
    """
    if y_axis is not None:
        if y_axis not in (0, 1, 2):
            raise ValueError("y_axis must be 0 (price), 1 (time), or 2 (cost)")
        return MERGED_Y_AXIS_NAMES[y_axis], y_axis
    if target_name not in MERGED_Y_AXIS_NAMES:
        raise ValueError(
            f"target_name must be one of {list(MERGED_Y_AXIS_NAMES)}; got {target_name!r}. "
            "Use y_axis=0,1,2 if you prefer numeric labels."
        )
    return target_name, MERGED_Y_AXIS_NAMES.index(target_name)


def _resolve_ve_only_graph(
    ve_only_graph: Optional[bool],
    exclude_static: bool,
    exclude_symbolic: bool,
) -> bool:
    """
    If ``ve_only_graph`` is not None, use it. Otherwise, when both program-feature
    exclusions are on (``no_pf``), default to |V|/|E|-only graph columns.
    """
    if ve_only_graph is not None:
        return bool(ve_only_graph)
    return bool(exclude_static and exclude_symbolic)


def _row_keep_mask(
    feature_names_x: List[str],
    *,
    exclude_static: bool,
    exclude_symbolic: bool,
    ve_only_graph: bool = False,
    graph_e_only: bool = False,
) -> np.ndarray:
    """
    Merged *input* columns (indices 3.. in the 53-d vector) follow FeatureMerger order:
    static_*, sym_*, graph_*/partition_*, conf_*.
    `exclude_static` drops static (SPF) program features; `exclude_symbolic` drops
    graph-parameterized symbolic (SGF) features.
    If ``graph_e_only`` is True, keep only :data:`GRAPH_NUM_EDGES_NAME` among
    graph_/partition_ columns (|E| only). If False and ``ve_only_graph`` is True,
    keep :data:`VE_ONLY_GRAPH_NAMES` (|V| and |E|). All ``conf_*`` are kept.
    """
    keep: List[bool] = []
    for name in feature_names_x:
        if name.startswith("static_"):
            keep.append(not exclude_static)
        elif name.startswith("sym_"):
            keep.append(not exclude_symbolic)
        elif name.startswith("graph_") or name.startswith("partition_"):
            if graph_e_only:
                keep.append(name == GRAPH_NUM_EDGES_NAME)
            elif ve_only_graph:
                keep.append(name in VE_ONLY_GRAPH_NAMES)
            else:
                keep.append(True)
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
    ve_only_graph: bool = False,
    graph_e_only: bool = False,
) -> MergedFeatureTable:
    """
    Return a new table with columns removed according to name prefixes; ``y`` unchanged.
    If ``graph_e_only`` is True, drop all ``graph_*`` / ``partition_*`` except
    ``graph_num_edges``. If False and ``ve_only_graph`` is True, keep only
    ``graph_num_vertices`` and ``graph_num_edges``.
    """
    if (
        not exclude_static
        and not exclude_symbolic
        and not ve_only_graph
        and not graph_e_only
    ):
        return table
    mask = _row_keep_mask(
        table.feature_names_x,
        exclude_static=exclude_static,
        exclude_symbolic=exclude_symbolic,
        ve_only_graph=ve_only_graph,
        graph_e_only=graph_e_only,
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


def apply_conf_batch_size_override(
    table: MergedFeatureTable,
    batch_size: Optional[float],
) -> MergedFeatureTable:
    """
    Replace the ``conf_batch_size`` column in ``X`` with ``batch_size`` for every row.

    Use after :func:`apply_merged_feature_exclusions` so the column is still present
    (it is a ``conf_*`` feature). If ``batch_size`` is None, return ``table`` unchanged.
    """
    if batch_size is None:
        return table
    if CONF_BATCH_SIZE_NAME not in table.feature_names_x:
        raise ValueError(
            f"batch_size override requires {CONF_BATCH_SIZE_NAME!r} in feature columns; "
            f"got {table.feature_names_x!r}"
        )
    j = table.feature_names_x.index(CONF_BATCH_SIZE_NAME)
    X = table.X.copy()
    X[:, j] = float(batch_size)
    return MergedFeatureTable(
        X=np.ascontiguousarray(X, dtype=np.float64),
        y=table.y,
        feature_names_x=table.feature_names_x,
        target_name=table.target_name,
        source_files=table.source_files,
    )


def load_merged_feature_dir(
    data_dir: str | Path,
    *,
    pattern: str = "*.yaml",
    target_name: str = _DEFAULT_TARGET,
    y_axis: Optional[int] = None,
) -> MergedFeatureTable:
    """
    Load all ``pattern`` files under ``data_dir``. Each file must have matching
    ``feature_names`` and ``feature_vector``.

    **Y** is the column at index ``y_axis`` (0 / 1 / 2) among ``price``,
    ``time``, ``cost``, or the column named by ``target_name`` when ``y_axis`` is
    None. **X** is always **columns 4..end** in file order (0-based index ``3`` onward),
    never the other two of ``price``/``time``/``cost``.

    The first three feature names must be exactly ``price``, ``time``, ``cost``.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    paths = sorted(root.glob(pattern), key=lambda p: p.name)
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} under {root}")

    ref_names: Optional[List[str]] = None
    x_name_idx: Optional[List[int]] = None
    target_idx: Optional[int] = None
    resolved_target_name: Optional[str] = None
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
        if len(names) < MERGED_X_START_INDEX + 1:
            raise ValueError(
                f"Need at least 4 feature names (3 label slots + 1+ inputs) in {p}"
            )
        if len(names) != len(vec):
            raise ValueError(f"feature_names and feature_vector length differ in {p}")

        row_names = [str(x) for x in names]
        if ref_names is None:
            ref_names = row_names
            if list(ref_names[:3]) != list(MERGED_Y_AXIS_NAMES):
                raise ValueError(
                    f"feature_names must start with {list(MERGED_Y_AXIS_NAMES)}; got {ref_names[:3]!r} in {p}"
                )
            resolved_target_name, target_idx = _resolve_merged_y(target_name, y_axis)
            if ref_names[target_idx] != resolved_target_name:
                raise ValueError("internal: target index mismatch for merged Y")
            x_name_idx = list(range(MERGED_X_START_INDEX, len(ref_names)))
        else:
            if row_names != ref_names:
                raise ValueError(
                    f"feature_names differ from other files: {p} vs first file"
                )

        v = _as_floats(vec)
        assert x_name_idx is not None and ref_names is not None and target_idx is not None
        assert resolved_target_name is not None
        y_col.append(v[target_idx])
        X_rows.append([v[i] for i in x_name_idx])
        if len(v) != len(ref_names):
            raise ValueError(f"Vector length != num names in {p}")

    assert ref_names is not None and x_name_idx is not None
    assert resolved_target_name is not None
    feature_names_x = [ref_names[i] for i in x_name_idx]

    return MergedFeatureTable(
        X=np.array(X_rows, dtype=np.float64),
        y=np.array(y_col, dtype=np.float64),
        feature_names_x=feature_names_x,
        target_name=resolved_target_name,
        source_files=[str(p) for p in paths],
    )


def _model_options_sanitized_for_meta(options: dict[str, Any]) -> dict[str, Any]:
    """JSON/YAML-friendly copy (tuples → lists, numpy scalars → Python)."""

    def _v(x: Any) -> Any:
        if x is None or isinstance(x, (bool, int, float, str)):
            return x
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            return x.item()  # type: ignore[union-attr]
        if isinstance(x, (tuple, list)):
            return [_v(i) for i in x]
        if isinstance(x, dict):
            return {str(ik): _v(iv) for ik, iv in x.items()}
        return str(x)

    out: dict[str, Any] = {}
    for k, v in options.items():
        out[str(k)] = _v(v)
    return out


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
    target_name: str = _DEFAULT_TARGET,
    y_axis: Optional[int] = None,
    n_iter: int = 300,
    exclude_static: bool = False,
    exclude_symbolic: bool = False,
    ve_only_graph: Optional[bool] = None,
    graph_e_only: bool = False,
    conf_batch_size: Optional[float] = None,
    model_kind: str = "bayesian",
    model_options: Optional[dict[str, Any]] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Fit a :class:`MergedTabularRegressor` (default: Bayesian) on merged YAMLs; save
    pickle + sidecar ``{model_basename}_meta.yaml`` (feature list, paths, ``model_kind``,
    metrics). Use ``create_model`` / :func:`register_model_kind` to plug in MLP, RL, etc.

    All paths are taken from the caller; nothing is hardcoded to a project root.

    **Y** is ``price`` / ``time`` / ``cost`` (see ``target_name`` or ``y_axis``).
    **X** is always features 4..end in each YAML (not the other two label columns).
    If ``y_axis`` is 0, 1, or 2, it overrides ``target_name`` for the label column.

    If ``exclude_static`` is True, drop ``static_*`` (static program) features. If
    ``exclude_symbolic`` is True, drop ``sym_*`` (graph-parameterized symbolic) features.
    If ``ve_only_graph`` is True, keep only ``graph_num_vertices`` and ``graph_num_edges``
    among ``graph_*`` / ``partition_*`` (diameter, skew, partition stats, etc. are
    dropped); ``static_*``, ``sym_*``, and ``conf_*`` stay. Use this for the
    **no_graph** ablation (CLI ``--graph-ve-only`` with no program-feature excludes).
    If ``ve_only_graph`` is None and both program-feature excludes are True, the same
    |V|/|E| graph slice applies (``no_pf`` default). Pass ``ve_only_graph=False`` to
    keep the full graph/partition block with ``no_pf`` (legacy).
    If ``graph_e_only`` is True, among graph/ partition columns keep only
    ``graph_num_edges`` (``no_all`` uses this with no SPF/SGF: |E| + conf only).
    For ``model_kind="bayesian"``, ``n_iter`` is the maximum number of variational
    / coordinate-ascent steps (passed through as ``n_iter`` to the Bayesian factory
    if not set in ``model_options``). Other kinds ignore the top-level ``n_iter`` unless
    you pass the relevant keys inside ``model_options`` (e.g. ``max_iter`` for MLP).
    If ``conf_batch_size`` is set, overwrites the ``conf_batch_size`` feature column
    for every training sample after feature exclusions (sweep config batch as one value).
    """
    _, y_axis_stored = _resolve_merged_y(target_name, y_axis)
    vog = _resolve_ve_only_graph(
        ve_only_graph, exclude_static, exclude_symbolic
    )
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = load_merged_feature_dir(
        data_dir, pattern=pattern, target_name=target_name, y_axis=y_axis
    )
    table = apply_merged_feature_exclusions(
        table,
        exclude_static=exclude_static,
        exclude_symbolic=exclude_symbolic,
        ve_only_graph=vog,
        graph_e_only=graph_e_only,
    )
    table = apply_conf_batch_size_override(table, conf_batch_size)
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

    mk = (model_kind or "bayesian").lower().strip()
    mo: dict[str, Any] = dict(model_options or {})
    if mk == "bayesian" and "n_iter" not in mo:
        mo["n_iter"] = n_iter
    model = create_model(mk, **mo)
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
        "ve_only_graph": bool(vog),
        "graph_e_only": bool(graph_e_only),
        "conf_batch_size_override": conf_batch_size,
        "n_iter": int(n_iter),
        "model_kind": mk,
        "model_options": _model_options_sanitized_for_meta(mo),
        "y_axis": y_axis_stored,
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
) -> Tuple[MergedTabularRegressor, dict[str, Any]]:
    """Load model saved by :func:`train_bayesian_cost_from_merged_yamls` and its ``*_meta.yaml``."""
    model_path = Path(model_path)
    meta_path = model_path.parent / f"{model_path.stem}_meta.yaml"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Expected meta YAML next to model: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    model = load_merged_regressor_file(str(model_path), meta)
    return model, meta


def optional_meta_for_model(model_path: Path) -> Optional[dict[str, Any]]:
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
    target_name: str = _DEFAULT_TARGET,
    y_axis: Optional[int] = None,
    check_feature_names: bool = True,
    conf_batch_size: Optional[float] = None,
) -> dict[str, Any]:
    """
    Load a :class:`MergedTabularRegressor` and merged YAMLs under ``data_dir`` (same layout
    as training). Return aggregate metrics and per-file predictions vs the label column
    (``target_name`` or ``y_axis``, same as :func:`load_merged_feature_dir`).

    If ``<stem>_meta.yaml`` exists next to the ``.pkl``, input feature names are
    compared when ``check_feature_names`` is True. Otherwise only the feature
    dimension must match the trained model input size.

    If ``conf_batch_size`` is not None, overwrites the ``conf_batch_size`` column for
    every row (match training with :func:`train_bayesian_cost_from_merged_yamls` and the
    same value). If None, uses each YAML's scalar as loaded (training meta may still
    list ``conf_batch_size_override`` for your records only).
    """
    model_path = Path(model_path)
    data_dir = Path(data_dir)

    meta0 = optional_meta_for_model(model_path)
    model = load_merged_regressor_file(str(model_path), meta0)
    if not model.is_fitted or model.n_input_features == 0:
        raise ValueError(f"Model is not fitted: {model_path}")

    meta = meta0
    table = load_merged_feature_dir(
        data_dir, pattern=pattern, target_name=target_name, y_axis=y_axis
    )
    ex_s = bool(meta.get("exclude_static", False)) if meta else False
    ex_y = bool(meta.get("exclude_symbolic", False)) if meta else False
    vog = bool(meta.get("ve_only_graph", False)) if meta else False
    g_e = bool(meta.get("graph_e_only", False)) if meta else False
    table = apply_merged_feature_exclusions(
        table,
        exclude_static=ex_s,
        exclude_symbolic=ex_y,
        ve_only_graph=vog,
        graph_e_only=g_e,
    )
    table = apply_conf_batch_size_override(table, conf_batch_size)
    n_in = int(table.X.shape[1])
    n_w = int(model.n_input_features)
    if n_in != n_w:
        raise ValueError(
            f"Input feature count {n_in} != model input size {n_w} "
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


def meta_exclusion_kwargs(meta: dict[str, Any] | None) -> dict[str, bool]:
    """Flags for :func:`apply_merged_feature_exclusions`, usually from ``*_meta.yaml``."""
    if not meta:
        return {
            "exclude_static": False,
            "exclude_symbolic": False,
            "ve_only_graph": False,
            "graph_e_only": False,
        }
    return {
        "exclude_static": bool(meta.get("exclude_static", False)),
        "exclude_symbolic": bool(meta.get("exclude_symbolic", False)),
        "ve_only_graph": bool(meta.get("ve_only_graph", False)),
        "graph_e_only": bool(meta.get("graph_e_only", False)),
    }


def x_row_from_merged_doc(
    doc: dict[str, Any],
    meta: dict[str, Any] | None,
    *,
    conf_batch_size: float | None = None,
) -> np.ndarray:
    """
    Build a single model-input row ``(1, d)`` from a merged feature YAML **document**
    (keys ``feature_names`` and ``feature_vector`` in 53-d layout). Applies the same
    static/symbolic/graph slicing and ``conf_batch_size`` override as training/eval.
    """
    names = doc.get("feature_names")
    vec = doc.get("feature_vector")
    if not isinstance(names, list) or not isinstance(vec, list):
        raise ValueError("doc must contain feature_names and feature_vector lists")
    row_names = [str(x) for x in names]
    v = _as_floats(vec)
    if len(row_names) < MERGED_X_START_INDEX + 1:
        raise ValueError("Need at least 4 features (3 labels + 1+ inputs)")
    if len(row_names) != len(v):
        raise ValueError("feature_names and feature_vector length differ")
    if list(row_names[:3]) != list(MERGED_Y_AXIS_NAMES):
        raise ValueError(
            f"feature_names must start with {list(MERGED_Y_AXIS_NAMES)}; got {row_names[:3]!r}"
        )
    x_idx = list(range(MERGED_X_START_INDEX, len(row_names)))
    X = np.array([[v[i] for i in x_idx]], dtype=np.float64)
    feature_names_x = [row_names[i] for i in x_idx]
    ex = meta_exclusion_kwargs(meta)
    table = MergedFeatureTable(
        X=X,
        y=np.array([0.0], dtype=np.float64),
        feature_names_x=feature_names_x,
        target_name="time",
        source_files=["<merged_doc>"],
    )
    table = apply_merged_feature_exclusions(
        table,
        exclude_static=ex["exclude_static"],
        exclude_symbolic=ex["exclude_symbolic"],
        ve_only_graph=ex["ve_only_graph"],
        graph_e_only=ex["graph_e_only"],
    )
    table = apply_conf_batch_size_override(table, conf_batch_size)
    return table.X


def x_matrix_from_merged_docs(
    docs: Sequence[dict[str, Any]],
    meta: dict[str, Any] | None,
    *,
    conf_batch_size: float | None = None,
) -> np.ndarray:
    """Stack :func:`x_row_from_merged_doc` for many YAML dicts (same layout each)."""
    if not docs:
        raise ValueError("docs is empty")
    parts: list[np.ndarray] = [
        x_row_from_merged_doc(d, meta, conf_batch_size=conf_batch_size) for d in docs
    ]
    d0 = int(parts[0].shape[1])
    for p in parts[1:]:
        if int(p.shape[1]) != d0:
            raise ValueError(
                f"Inconsistent X width after preprocessing: {d0} vs {int(p.shape[1])}"
            )
    return np.vstack(parts)
