# AutoConfig API reference

## Module map

| Module | Role |
|--------|------|
| `autoconfig` | Public exports |
| `autoconfig.feature_extractor` | Static / graph / config feature code |
| `autoconfig.models` | Bayesian and related models |
| `autoconfig.prediction` | Optional code-path predictor (`CostPredictor` + `FeatureManager`) |
| `autoconfig.merged` | Merged 53-D YAML training + `bayesian_cost_merged.pkl` inference |

---

## MergedBayesianPredictor

**Module:** `autoconfig.merged.predictor`

Loads `bayesian_cost_merged.pkl` and sibling `*_meta.yaml` produced by `train-merged`. Use for **inference** on merged-feature rows (same `feature_names_x` as training).

### `MergedBayesianPredictor.load(path_to_pkl)`

```python
from autoconfig import MergedBayesianPredictor
import numpy as np

pred = MergedBayesianPredictor.load("out/models/bayesian_cost_merged.pkl")
# Aligned X only (n × n_features), same order as meta["feature_names_x"]
y = pred.predict_batch(X)
y0 = pred.predict_one(X[0])

# From one merged YAML dict (keys feature_names, feature_vector)
y_hat = pred.predict_merged_doc(doc)
y_batch = pred.predict_merged_docs([doc1, doc2, doc3])
```

### `predict_one` / `predict_batch` / `predict_merged_doc` / `predict_merged_docs`

See module docstring. `predict_merged_doc*` apply the same static/symbolic/graph exclusions as training (from meta).

---

## CostPredictor (code path)

**Module:** `autoconfig.prediction.cost_predictor`

Predicts graph-query execution time from query **code string** + data graph + config via `FeatureManager` (not the merged-YAML model).

### `__init__(model=None)`

```python
from autoconfig.prediction import CostPredictor
from autoconfig import BayesianExecutionTimeModel

predictor = CostPredictor()
predictor = CostPredictor(model=BayesianExecutionTimeModel(n_iter=500))
```

### `train(queries, graphs, configs, execution_times, verbose=False)`

**Parameters**

- `queries` — list of `nx.Graph` or dicts
- `graphs` — list of data graphs
- `configs` — list of configuration dicts
- `execution_times` — `ndarray` of measured times
- `verbose` — print progress

**Returns:** dict with metrics (`mae`, `rmse`, `mape`, `r2`, …).

### `predict(query, graph, config, return_uncertainty=False)`

Point prediction or `(pred, lower, upper)` if `return_uncertainty=True`.

### `predict_batch(queries, graphs, configs, return_uncertainty=False)`

Vectorized predictions.

### `evaluate(queries, graphs, configs, execution_times)`

Returns metric dict on a holdout set.

### `save_model` / `load_model`

Persist or restore predictor state.

### `get_feature_importance` / `get_feature_names`

Inspection helpers.

---

## FeatureManager

**Module:** `autoconfig.feature_extractor.feature_manager`

### `extract_all(query_graph, data_graph, config)`

Returns a single concatenated feature vector from NetworkX graphs.

### `extract_all_from_dict(query_dict, graph_dict, config)`

Same, from `{'nodes': [...], 'edges': [...]}`-style dicts.

### `get_feature_names()`

### `get_feature_dimensions()`

Returns `(query_dim, graph_dim, config_dim)`.

### `normalize_features(features, means=None, stds=None)`

Z-score normalization helper.

---

## QueryFeatureExtractor (pattern graph)

**Module:** `autoconfig.feature_extractor.query_extractor`

### `extract(query_graph)` → 13-D vector

Names include: `query_num_nodes`, `query_num_edges`, `query_density`, `query_avg_degree`, `query_max_degree`, `query_min_degree`, `query_std_degree`, `query_num_triangles`, `query_clustering_coeff`, `query_diameter`, `query_is_connected`, `query_num_components`, `query_avg_path_length`.

### `extract_from_dict`, `get_feature_names`

---

## Graph features YAML for merge

The CLI no longer builds `graph_features.yaml` from edge lists. Author that file (or generate it with your own tooling) so it matches the schema expected by :class:`autoconfig.utils.feature_merger.FeatureMerger` — see [feature_extraction.md](feature_extraction.md) and [CONFIG_GUIDE.md](CONFIG_GUIDE.md).

**Internal graph stats for the legacy `FeatureManager` code path** still use :class:`autoconfig.feature_extractor.graph_partition_extractor.GraphPartitionExtractor` (not removed).

---

## ConfigFeatureExtractor

**Module:** `autoconfig.feature_extractor.config_extractor`

### `extract(config)` → 10-D vector (legacy extractor path)

**`FeatureMerger`** (class `autoconfig.utils.feature_merger`) builds a **53-D** row: **`price`**, **`time`**, **`cost`**, then static (8), symbolic (12), graph/partition (17), and config (**13** scalars: twelve `conf_*` fields + **`conf_price`**). See [feature_extraction.md](feature_extraction.md) for the full name list.

**Training / eval on merged YAML directories** — module `autoconfig.merged` (`pipeline`): `load_merged_feature_dir`, `train_bayesian_cost_from_merged_yamls`, `evaluate_bayesian_cost_on_merged_dir`. **X** uses **indices 3…** (features 4–53); **Y** is selected by **integer `y_axis` ∈ {0,1,2}** (CLI **`-y`**) with names in `MERGED_Y_AXIS_NAMES`. The Python API also accepts `target_name` for the same three columns. Constants: `MERGED_Y_AXIS_NAMES`, `MERGED_X_START_INDEX`. Mapping: [TRAIN_TEST_MERGED.md](TRAIN_TEST_MERGED.md#label-y-in-the-cli--y----y-axis).

---

## BayesianExecutionTimeModel

**Module:** `autoconfig.models.bayesian_model`

### Constructor hyperparameters

- `alpha_1`, `alpha_2` — noise precision priors
- `lambda_1`, `lambda_2` — weight precision priors
- `n_iter` — max iterations (default 300)
- `tol` — convergence tolerance
- `use_log_transform` — log-transform targets

### `fit(X, y, verbose=False)`

### `predict(X, return_std=False)`

### `predict_with_uncertainty(X, confidence_level=0.95)`

### `score(X, y)` — R²

### `get_params` / `set_params`, `save` / `load`

---

## Data structures

### Query as dict

```python
query_dict = {
    "nodes": [0, 1, 2, 3],
    "edges": [(0, 1), (1, 2), (2, 3)],
}
```

### Config dict (illustrative)

```python
config = {
    "memory_limit": int,       # MB
    "num_threads": int,
    "cache_size": int,
    "batch_size": int,
    "io_buffer_size": int,
    "num_workers": int,
    "timeout": int,
    "enable_index": bool,
    "index_type": str,
    "compression_enabled": bool,
}
```

---

## Error handling

```python
from autoconfig.prediction import CostPredictor
predictor = CostPredictor()
try:
    predictor.predict(query, graph, config)
except ValueError as e:
    ...

try:
    predictor.load_model("missing.pkl")
except FileNotFoundError:
    ...
```
