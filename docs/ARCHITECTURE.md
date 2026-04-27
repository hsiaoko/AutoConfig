# AutoConfig architecture

## Overview

The package is a **set of tools** (not “online vs offline” packages):

1. **Feature extraction** — query / graph / config via `autoconfig.feature_extractor` and `autoconfig.utils` extractors; **merge** into 53-D rows via `FeatureMerger`.
2. **Config YAML generation** — separate scripts (e.g. `data/conf/build_ten_conf.py` LHS), not required imports from training code.
3. **Merged 53-D training and eval** — `autoconfig.merged` (`train-merged` / `eval-merged` CLI; `BayesianCostModel` on disk).
4. **Inference** — `MergedBayesianPredictor` for `bayesian_cost_merged.pkl` + `*_meta.yaml`. Optional: `autoconfig.prediction.CostPredictor` (`FeatureManager` + `BayesianExecutionTimeModel`) for raw Q,G,C code paths.

## System diagram

```
┌────────────────────────────────────────────────────────────────┐
│  Feature extraction: query, graph, config  →  merge (53-D YAML)  │
├────────────────────────────────────────────────────────────────┤
│  autoconfig.merged: load YAML tables → train → .pkl + _meta   │
│  MergedBayesianPredictor: single/batch / merged-doc inference   │
├────────────────────────────────────────────────────────────────┤
│  (optional) autoconfig.prediction.CostPredictor: Q,G,C code API │
└────────────────────────────────────────────────────────────────┘
```

## Repository layout (excerpt)

```
autoconfig/
├── merged/           # train/eval on merged YAMLs; MergedBayesianPredictor
├── feature_extractor/
├── models/           # BayesianExecutionTimeModel, BayesianCostModel, …
├── utils/            # query/graph extractors, FeatureMerger, …
└── prediction/       # optional CostPredictor (FeatureManager path)
data/conf/            # e.g. build_ten_conf.py (LHS) — scripts, not a Python subpackage
```

## Python API (sketch)

**Merged model (primary):**

```python
from autoconfig.merged import (
    train_bayesian_cost_from_merged_yamls,
    evaluate_bayesian_cost_on_merged_dir,
)
from autoconfig import MergedBayesianPredictor

pred = MergedBayesianPredictor.load("out/models/bayesian_cost_merged.pkl")
# pred.predict_one(X_row), pred.predict_merged_doc(yaml_dict), …
```

**Optional code-path predictor:**

```python
from autoconfig.prediction import CostPredictor
# predictor.train(..., FeatureManager feature matrices)
```

## CLI summary

```bash
autoconfig query / graph / merge / all
autoconfig train-merged …
autoconfig eval-merged …
```

## API reference

See [api_reference.md](api_reference.md).
