# AutoConfig API reference

## Module map

| Module | Role |
|--------|------|
| `autoconfig` | Public exports |
| `autoconfig.feature_extractor` | Static / graph / config feature code |
| `autoconfig.models` | Bayesian and related models |
| `autoconfig.prediction` | High-level predictors |

---

## CostPredictor

**Module:** `autoconfig.prediction.cost_predictor`

Predicts graph-query execution time from query graph + data graph + config.

### `__init__(model=None)`

```python
from autoconfig import CostPredictor, BayesianExecutionTimeModel

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

## GraphFeatureExtractor

**Module:** `autoconfig.feature_extractor.graph_extractor`

### `extract(data_graph)` → 15-D vector

Adds assortativity and transitivity on top of structural stats similar to the query extractor.

---

## ConfigFeatureExtractor

**Module:** `autoconfig.feature_extractor.config_extractor`

### `extract(config)` → 10-D vector

Typical names: `conf_memory_limit`, `conf_num_threads`, `conf_cache_size`, `conf_batch_size`, `conf_io_buffer_size`, `conf_num_workers`, `conf_timeout`, `conf_enable_index`, `conf_index_type`, `conf_compression_enabled`.

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
