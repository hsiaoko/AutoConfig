"""Write recommended configurations to ``data/conf``-style YAML files (one file per config)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .recommend import ConfRecommendResult

_DEFAULT_CATALOG: list[dict[str, Any]] = [{"price": 0.0}]


def export_recommendations_to_dir(
    result: ConfRecommendResult,
    out_dir: str | Path,
    *,
    prefix: str = "recommend",
    also_write_full_report: bool = False,
) -> list[Path]:
    """
    Create ``out_dir`` and write ``{prefix}_rank{NN}.yaml`` for each :attr:`result.items` row.
    Each file has ``metadata``, ``catalog`` (from result or a default), and a single
    ``configurations:`` list with one object — same family as ``data/conf/gpu/*.yaml``.

    If ``also_write_full_report`` is True, also writes ``{prefix}_full_report.yaml`` with
    :meth:`ConfRecommendResult.to_dict`.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    catalog: list[dict[str, Any]] = copy.deepcopy(
        result.catalog if result.catalog is not None else _DEFAULT_CATALOG
    )
    written: list[Path] = []

    for it in result.items:
        doc: dict[str, Any] = {
            "metadata": {
                "autoconfig_recommend": True,
                "rank": it.rank,
                "predicted": float(it.predicted),
                "config_index": it.config_index,
                "target_name": result.target_name,
                "y_axis": result.y_axis,
                "model_path": result.model_path,
            },
            "catalog": copy.deepcopy(catalog),
            "configurations": [copy.deepcopy(it.configuration)],
        }
        if it.source_path is not None:
            doc["metadata"]["source_path"] = it.source_path
        if result.refine_iterations:
            doc["metadata"]["refine_iterations"] = result.refine_iterations
        if result.refine_stopped_reason is not None:
            doc["metadata"]["refine_stopped_reason"] = result.refine_stopped_reason

        name = f"{prefix}_rank{it.rank:02d}.yaml"
        p = out / name
        p.write_text(
            yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        written.append(p)

    if also_write_full_report:
        rep = out / f"{prefix}_full_report.yaml"
        rep.write_text(
            yaml.safe_dump(
                result.to_dict(),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        written.append(rep)

    return written


def write_recommendation_report(
    result: ConfRecommendResult,
    filepath: str | Path,
) -> Path:
    """Write :meth:`ConfRecommendResult.to_dict` to a single YAML file (legacy / summary)."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        yaml.safe_dump(
            result.to_dict(),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return p
