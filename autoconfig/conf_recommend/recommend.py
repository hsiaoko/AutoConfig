"""
Rank configuration candidates with a trained ``train-merged`` regressor (cost / time / price).

Builds 53-d merged feature rows from **one** query (task) + **one** graph + **many** configs
via :class:`autoconfig.utils.feature_merger.FeatureMerger`, then runs
:class:`autoconfig.merged.predictor.MergedBayesianPredictor` batch predict.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import yaml

from ..merged import MERGED_Y_AXIS_NAMES
from ..merged.predictor import MergedBayesianPredictor
from ..utils.feature_merger import FeatureMerger

# Accept on-disk feature YAML or an in-memory dict (same structure as :meth:`FeatureMerger.load_yaml`).
PathLike = Union[str, Path]
QueryOrGraph = Union[Mapping[str, Any], PathLike]
ConfigInput = Union[Mapping[str, Any], PathLike, Sequence[Mapping[str, Any]]]


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.unsafe_load(f)


def _as_dict(data: Any, name: str) -> dict[str, Any]:
    if isinstance(data, dict):
        return dict(data)
    p = Path(data)
    if p.is_dir():
        raise IsADirectoryError(
            f"{name} is a directory; for query/graph use a file path. "
            f"For config, pass it as `config_candidates` to `recommend_top_k` (supported there)."
        )
    if not p.is_file():
        raise FileNotFoundError(f"{name} must be a dict or an existing file path, got {data!r}")
    return _load_yaml(p)


def _load_config_dir(
    path: Path,
) -> tuple[dict[str, Any], list[str]]:
    """
    One `*.yaml` / `*.yml` per file: concatenate ``configurations`` (in sorted filename order).

    * Standard layout: each file is like a single-``configurations`` system config
      (``catalog`` + ``configurations:``; see ``data/conf/gpu/conf_01.yaml``).
    * A file may also be a **bare** single row (``node_id`` / ``resource`` / …) with no
      ``configurations:`` key — it is treated as one candidate.

    The first file that contains ``catalog`` supplies it for :meth:`FeatureMerger.extract_config_features`
    (unless the caller passes ``catalog=`` to :func:`recommend_top_k`). Rows remember ``source_path``.
    """
    pats = sorted(
        p
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in (".yaml", ".yml") and not p.name.startswith(".")
    )
    if not pats:
        raise ValueError(
            f"No .yaml or .yml found under {path} (one configuration per file, or per-file list)"
        )
    catalog: Optional[list[dict[str, Any]]] = None
    configurations: list[dict[str, Any]] = []
    source_paths: list[str] = []

    for fp in pats:
        d = _load_yaml(fp)
        if not isinstance(d, dict):
            raise TypeError(f"Expected a mapping in {fp}, got {type(d)}")

        if catalog is None and d.get("catalog"):
            catalog = [dict(x) for x in d["catalog"] if isinstance(x, dict)]

        confs = d.get("configurations")
        if confs is not None:
            if not isinstance(confs, list):
                raise TypeError(f"'configurations' in {fp} must be a list")
            for c in confs:
                if not isinstance(c, dict):
                    continue
                configurations.append(dict(c))
                source_paths.append(str(fp.resolve()))
        else:
            if "graph_features" in d or "query_features" in d:
                raise ValueError(
                    f"{fp} looks like a full merged/query yaml, not a system config. "
                    "Use one conf layout per file (``configurations:`` or bare node/resource block)."
                )
            if d.get("resource") is not None or d.get("node_id") is not None or d.get("k") is not None:
                configurations.append(dict(d))
                source_paths.append(str(fp.resolve()))
            else:
                raise ValueError(
                    f"{fp} has no 'configurations' list; add it, or use a bare object with at least "
                    "'resource' or 'node_id' (one configuration in this file)."
                )

    if not configurations:
        raise ValueError(f"No configuration rows under {path}")
    if catalog is None:
        catalog = [{"price": 0.0}]

    return (
        {
            "configurations": configurations,
            "catalog": catalog,
        },
        source_paths,
    )


def _normalize_config_input(
    config_candidates: ConfigInput,
    catalog: Optional[Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], Optional[list[str]]]:
    """
    Build a `config_data` dict for :meth:`FeatureMerger.extract_config_features`.

    Returns ``(config_data, source_paths)`` where ``source_paths`` is one path string
    per configuration row when **config_candidates** is a **directory** of yaml files, else
    ``None`` (unknown provenance for single-file / list / dict).
    """
    if isinstance(config_candidates, (str, Path)):
        p = Path(config_candidates)
        if p.is_dir():
            cdata, source_paths = _load_config_dir(p)
            if catalog is not None:
                cdata["catalog"] = [dict(x) for x in catalog]
            return cdata, source_paths
        d = _load_yaml(p)
        if not isinstance(d, dict):
            raise TypeError("config YAML root must be a mapping")
        if catalog is not None:
            d = dict(d)
            d["catalog"] = [dict(x) for x in catalog]
        return d, None

    if isinstance(config_candidates, dict):
        if "configurations" in config_candidates or "config_features" in config_candidates:
            out = dict(config_candidates)
            if catalog is not None:
                out["catalog"] = [dict(x) for x in catalog]
            return out, None
        raise ValueError(
            "config dict must contain 'configurations' and optional 'catalog', "
            "or 'config_features' (pre-extracted rows). "
            "Or pass a list of configuration objects, or a directory of per-file yamls."
        )

    if isinstance(config_candidates, (list, tuple)):
        rows = [dict(c) for c in config_candidates]
        if not rows:
            raise ValueError("config_candidates list is empty")
        out2: dict[str, Any] = {"configurations": rows}
        if catalog is not None:
            out2["catalog"] = [dict(x) for x in catalog]
        else:
            out2["catalog"] = [{"price": 0.0}]
        return out2, None

    raise TypeError(
        f"config_candidates must be path, dict, or list[dict], got {type(config_candidates)}"
    )


@dataclass
class ConfRecommendation:
    """One ranked candidate after prediction."""

    rank: int
    config_index: int
    predicted: float
    configuration: dict[str, Any] = field(repr=False)
    source_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "rank": self.rank,
            "config_index": self.config_index,
            "predicted": self.predicted,
            "configuration": self.configuration,
        }
        if self.source_path is not None:
            d["source_path"] = self.source_path
        return d


@dataclass
class ConfRecommendResult:
    """Outcome of a top-``k`` run (``target_name`` is what the model was trained to predict)."""

    model_path: str
    target_name: str
    y_axis: int
    k: int
    items: list[ConfRecommendation]
    lower_is_better: bool = True
    # Optional: local search after :func:`recommend_top_k`
    initial_items: Optional[list[ConfRecommendation]] = None
    refine_iterations: int = 0
    refine_stopped_reason: Optional[str] = None
    catalog: Optional[list[dict[str, Any]]] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model_path": self.model_path,
            "target_name": self.target_name,
            "y_axis": self.y_axis,
            "k": self.k,
            "lower_is_better": self.lower_is_better,
            "items": [x.to_dict() for x in self.items],
        }
        if self.initial_items is not None:
            d["initial_items"] = [x.to_dict() for x in self.initial_items]
        if self.refine_iterations:
            d["refine_iterations"] = self.refine_iterations
        if self.refine_stopped_reason is not None:
            d["refine_stopped_reason"] = self.refine_stopped_reason
        if self.catalog is not None:
            d["catalog"] = copy.deepcopy(self.catalog)
        return d


def _per_configuration_rows(
    config_data: dict[str, Any], cfg_feat_rows: list[dict[str, float]]
) -> list[dict[str, Any]]:
    n = len(cfg_feat_rows)
    confs = config_data.get("configurations")
    if confs is not None and len(confs) == n:
        return [dict(c) for c in confs]
    cfe = config_data.get("config_features")
    if cfe is not None and len(cfe) == n:
        return [dict(cfe[i]) for i in range(n)]
    return [dict(cfg_feat_rows[i]) for i in range(n)]


def _merged_docs_for_configs(
    merger: FeatureMerger,
    query_data: dict[str, Any],
    graph_data: dict[str, Any],
    config_data: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg_rows = merger.extract_config_features(config_data)
    matrix, feature_names = merger.merge(query_data, graph_data, cfg_rows)
    names_list = list(feature_names)
    docs: list[dict[str, Any]] = []
    for i in range(int(matrix.shape[0])):
        docs.append(
            {
                "feature_names": names_list,
                "feature_vector": [float(x) for x in matrix[i].tolist()],
            }
        )
    per_cfg = _per_configuration_rows(config_data, cfg_rows)
    return docs, per_cfg


def recommend_top_k(
    model_path: str | Path,
    query_features: QueryOrGraph,
    graph_features: QueryOrGraph,
    config_candidates: ConfigInput,
    *,
    k: int = 5,
    catalog: Optional[Sequence[Mapping[str, Any]]] = None,
    conf_batch_size: Optional[float] = None,
    lower_is_better: bool = True,
    local_refine: bool = False,
    refine_max_iterations: int = 10_000,
    refine_seed: Optional[int] = None,
) -> ConfRecommendResult:
    """
    Predict the trained target (e.g. ``time`` or ``cost``) for every merged row and return
    the **best** ``k`` candidates.

    If ``local_refine`` is True, after the initial top-``k``, repeatedly random-perturb those
    configurations (``cpu_cores`` ±16, ``memory_gb`` in ±16·N GB, GPU ``grid_size``/``block_size``
    ×2 or ÷2 when ``num_gpus``>0) and re-predict; keep improving until a round has no
    **strict** improvement, or ``refine_max_iterations`` (see :mod:`.refine_search`).

    * **query_features** — task (query) YAML or dict, ``query_features.static`` / ``symbolic`` as in training.
    * **graph_features** — one graph YAML or dict, ``graph_features`` block as in training.
    * **config_candidates** — (1) path to a **directory** of ``*.yaml`` (one file per
      configuration, same layout as ``data/conf/`` single-conf files; rows are joined in
      filename order, ``source_path`` is set in each :class:`ConfRecommendation`),
      (2) path to a single config YAML (multiple ``configurations`` allowed in-file),
      (3) a dict, or (4) a ``list[dict]`` of configuration objects; the list case uses a
      default ``catalog=[{price:0.0}]`` unless ``catalog`` is passed.

    The model is loaded with ``train-merged`` metadata (feature exclusions, etc.) applied
    the same way as :class:`MergedBayesianPredictor`.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    mp = Path(model_path)
    predictor = MergedBayesianPredictor.load(mp)
    meta = predictor.meta
    y_axis = int(meta.get("y_axis", 1))
    target_name = str(meta.get("target", MERGED_Y_AXIS_NAMES[y_axis]))
    if target_name not in MERGED_Y_AXIS_NAMES:
        target_name = MERGED_Y_AXIS_NAMES[y_axis]

    q = _as_dict(query_features, "query_features")
    g = _as_dict(graph_features, "graph_features")
    cdata, source_paths = _normalize_config_input(config_candidates, catalog)

    merger = FeatureMerger()
    docs, per_cfg = _merged_docs_for_configs(merger, q, g, cdata)
    n = len(docs)
    if n == 0:
        raise ValueError("No configuration rows after feature extraction")
    if k > n:
        k = n

    preds = predictor.predict_merged_docs(
        docs, conf_batch_size=conf_batch_size
    )
    y = np.asarray(preds, dtype=np.float64).ravel()
    if y.shape[0] != n:
        raise RuntimeError("prediction length does not match number of config rows")

    order = np.argsort(y) if lower_is_better else np.argsort(-y)
    best = order[:k].tolist()
    items: list[ConfRecommendation] = []
    for rank, idx in enumerate(best, start=1):
        j = int(idx)
        sp: Optional[str] = None
        if source_paths is not None and j < len(source_paths):
            sp = source_paths[j]
        items.append(
            ConfRecommendation(
                rank=rank,
                config_index=j,
                predicted=float(y[j]),
                configuration=per_cfg[j],
                source_path=sp,
            )
        )
    cat_snapshot: list[dict[str, Any]] = list(cdata.get("catalog") or [{"price": 0.0}])
    result = ConfRecommendResult(
        model_path=str(mp),
        target_name=target_name,
        y_axis=y_axis,
        k=len(items),
        items=items,
        lower_is_better=lower_is_better,
        catalog=cat_snapshot,
    )

    if not local_refine or refine_max_iterations < 1:
        return result

    from .refine_search import local_refine_top_k

    cat_list: list[dict[str, Any]] = list(
        cdata.get("catalog") or [{"price": 0.0}]
    )
    return local_refine_top_k(
        base=result,
        predictor=predictor,
        merger=merger,
        query_data=q,
        graph_data=g,
        catalog=cat_list,
        conf_batch_size=conf_batch_size,
        k=k,
        lower_is_better=lower_is_better,
        max_iterations=refine_max_iterations,
        seed=refine_seed,
    )
