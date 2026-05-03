"""
Local search: random perturbations of top-k configurations until model predictions stop improving.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..merged.predictor import MergedBayesianPredictor
from ..utils.feature_merger import FeatureMerger
from .recommend import (
    ConfRecommendation,
    ConfRecommendResult,
    _merged_docs_for_configs,
)


def perturb_configuration(
    configuration: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Random one-step perturbation:

    * ``resource.cpu_cores``: add integer in **[-16, 16]**, clamp to >= 1.
    * ``resource.memory_gb``: add **16 GB** times a small random integer in **[-3, 3]**
      (16 GB step, total shift up to ±48 GB).
    * If ``num_gpus`` > 0: ``grid_size`` / ``block_size`` (when >0) are each
      independently multiplied by 2 or integer-divided by 2 (min 1).
    """
    out = copy.deepcopy(configuration)
    res = out.get("resource")
    if not isinstance(res, dict):
        res = {}
        out["resource"] = res

    cc = int(round(float(res.get("cpu_cores", 4) or 4)))
    cc = max(1, cc + int(rng.integers(-16, 17)))
    res["cpu_cores"] = cc

    mem = float(res.get("memory_gb", 8) or 8)
    mem_steps = int(rng.integers(-3, 4))
    mem = max(1.0, mem + float(16 * mem_steps))
    res["memory_gb"] = mem

    ng = int(res.get("num_gpus", 0) or 0)
    gs = int(res.get("grid_size", 0) or 0)
    bs = int(res.get("block_size", 0) or 0)
    if ng > 0:
        if gs > 0:
            res["grid_size"] = max(1, gs * 2 if rng.random() < 0.5 else max(1, gs // 2))
        if bs > 0:
            res["block_size"] = max(1, bs * 2 if rng.random() < 0.5 else max(1, bs // 2))
    return out


def _predict_for_configurations(
    predictor: MergedBayesianPredictor,
    merger: FeatureMerger,
    query_data: dict[str, Any],
    graph_data: dict[str, Any],
    configurations: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
    conf_batch_size: Optional[float],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    cdata: dict[str, Any] = {
        "configurations": configurations,
        "catalog": [dict(x) for x in catalog],
    }
    docs, per_cfg = _merged_docs_for_configs(merger, query_data, graph_data, cdata)
    preds = predictor.predict_merged_docs(docs, conf_batch_size=conf_batch_size)
    y = np.asarray(preds, dtype=np.float64).ravel()
    if y.shape[0] != len(configurations):
        raise RuntimeError("refine: prediction length mismatch")
    return y, per_cfg


def _dedupe_top_k(
    evaluated: list[tuple[float, dict[str, Any]]],
    k: int,
    *,
    lower_is_better: bool,
) -> list[tuple[float, dict[str, Any]]]:
    evaluated = sorted(
        evaluated,
        key=lambda t: t[0],
        reverse=not lower_is_better,
    )
    seen: set[str] = set()
    out: list[tuple[float, dict[str, Any]]] = []
    for pred, cfg in evaluated:
        key = json.dumps(cfg, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append((float(pred), copy.deepcopy(cfg)))
        if len(out) >= k:
            break
    return out


def local_refine_top_k(
    *,
    base: ConfRecommendResult,
    predictor: MergedBayesianPredictor,
    merger: FeatureMerger,
    query_data: dict[str, Any],
    graph_data: dict[str, Any],
    catalog: Sequence[Mapping[str, Any]],
    conf_batch_size: Optional[float],
    k: int,
    lower_is_better: bool,
    max_iterations: int,
    seed: Optional[int],
) -> ConfRecommendResult:
    """
    Perturb each of the current top-``k`` configs per iteration; stop when the best
    prediction in a batch is not **strictly** better than the global best so far.
    """
    catalog_list = [dict(x) for x in catalog]
    rng = np.random.default_rng(seed)

    initial_items = [copy.deepcopy(x) for x in base.items]
    evaluated: list[tuple[float, dict[str, Any]]] = [
        (it.predicted, copy.deepcopy(it.configuration)) for it in base.items
    ]
    global_best = float(min(t[0] for t in evaluated)) if lower_is_better else float(
        max(t[0] for t in evaluated)
    )
    parents = [copy.deepcopy(it.configuration) for it in base.items]
    k_eff = len(parents)
    if k_eff == 0:
        return base

    reason = "no_improvement"
    iters_run = 0
    while iters_run < max_iterations:
        iters_run += 1
        children = [perturb_configuration(p, rng) for p in parents]
        preds, per_cfg = _predict_for_configurations(
            predictor,
            merger,
            query_data,
            graph_data,
            children,
            catalog_list,
            conf_batch_size,
        )
        round_extreme = float(np.min(preds)) if lower_is_better else float(np.max(preds))
        strict_better = (
            (round_extreme < global_best - 1e-12)
            if lower_is_better
            else (round_extreme > global_best + 1e-12)
        )
        if not strict_better:
            reason = "no_improvement"
            break

        for i in range(len(children)):
            evaluated.append((float(preds[i]), copy.deepcopy(per_cfg[i])))
        global_best = min(t[0] for t in evaluated) if lower_is_better else max(
            t[0] for t in evaluated
        )
        order = np.argsort(preds) if lower_is_better else np.argsort(-preds)
        parents = [copy.deepcopy(children[int(j)]) for j in order[:k_eff]]
    else:
        reason = "max_iterations"

    best_k = _dedupe_top_k(evaluated, k, lower_is_better=lower_is_better)
    final_items: list[ConfRecommendation] = []
    for rank, (pred, cfg) in enumerate(best_k, start=1):
        final_items.append(
            ConfRecommendation(
                rank=rank,
                config_index=rank - 1,
                predicted=pred,
                configuration=cfg,
                source_path=None,
            )
        )

    return ConfRecommendResult(
        model_path=base.model_path,
        target_name=base.target_name,
        y_axis=base.y_axis,
        k=len(final_items),
        items=final_items,
        lower_is_better=base.lower_is_better,
        initial_items=initial_items,
        refine_iterations=iters_run,
        refine_stopped_reason=reason,
        catalog=catalog_list,
    )
