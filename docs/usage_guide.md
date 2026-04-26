# AutoConfig usage guide

## Overview

End-to-end feature workflow:

1. **Query** — static + symbolic templates from source
2. **Graph** — structure and partition statistics
3. **Config** — author `config_features.yaml`, or use `data/conf/build_ten_conf.py` (LHS samples)
4. **Merge** — instantiate symbols and build the numeric matrix

```
query.yaml ────┐
graph.csv  ────┼──► merge ──► merged_features.yaml
configs.yaml ──┘
```

---

## CLI

### 1. Query features

```bash
autoconfig query --input <file> --output out/query_features.yaml
```

Example:

```bash
autoconfig query --input data/queries/kernel_bfs.cu --output out/query_features.yaml
```

### 2. Graph features

```bash
autoconfig graph --input data/edges.csv --output out/graph_features.yaml
# partitioned directory
autoconfig graph --input data/partitions/ --output out/graph_features.yaml
```

### 3. Configurations

Author a YAML for merge, or run `python data/conf/build_ten_conf.py` (see [CONFIG_GUIDE.md](CONFIG_GUIDE.md)).

### 4. Merge

```bash
autoconfig merge \
  --query out/query_features.yaml \
  --graph out/graph_features.yaml \
  --config out/config_features.yaml \
  --output out/merged_features.yaml
```

---

## Full pipeline example

```bash
autoconfig query --input data/queries/kernel_bfs.cu --output out/query.yaml
autoconfig graph --input data/graph_medium_pl.csv --output out/graph.yaml
# e.g. python data/conf/build_ten_conf.py -n 10  →  use one file as out/config.yaml
autoconfig merge -q out/query.yaml -g out/graph.yaml -c out/config.yaml -o out/merged.yaml
```

---

## Feature groups (merge output per group)

| Group | Count | Role |
|-------|-------|------|
| Static | 8 | Query structure |
| Symbolic | 12 | Templates → numeric at merge |
| Graph | 17 | Graph / partition |
| Config | 13 | Resources + `conf_grid_size` / `conf_block_size` + **`conf_price`** |
| **Subtotal (Φ block)** | **50** | |
| + `price`, `time`, `cost` | 3 | Leading scalars in **53-D** batch / `FeatureMerger` rows |
| **Total (training YAMLs)** | **53** | |

Symbolic instantiation examples (conceptual):

| Template | Placeholder | Uses graph stats |
|----------|-------------|------------------|
| VScan | 1.0 | \|V\| |
| EScan | 1.0 | \|E\| |
| FScan | 1.0 | diameter, \|E\|, … |
| RExp | 1.0 | avg degree, diameter |
| Atom | 1.0 | \|E\|, skew |
| Comm | 1.0 | boundary degrees |

---

## Training merged YAMLs (53-D)

For directories of merged feature files (`feature_names` / `feature_vector`, length **53**), use **`autoconfig train-merged`** and **`autoconfig eval-merged`**:

- **X (inputs):** features **4–53** in file order (static through `conf_price`) — the first three slots (`price`, `time`, `cost`) are **not** fed as inputs when predicting another label.
- **Y (label):** one of those first three, selected with **`--y-axis 0|1|2`** (price / time / cost) or **`--target price|time|cost`** (`--y-axis` overrides `--target` if both are set).

Full workflow, scripts, and metadata: **[TRAIN_TEST_MERGED.md](TRAIN_TEST_MERGED.md)**.

---

## Python API (extractors + merger)

```python
from autoconfig.utils.query_feature_extractor import QueryFeatureExtractor
from autoconfig.utils.graph_feature_extractor import GraphFeatureExtractor
from autoconfig.utils.feature_merger import FeatureMerger  # merge_all loads YAML paths

query_ext = QueryFeatureExtractor()
query_features = query_ext.extract_from_file("data/queries/kernel_bfs.cu")

graph_ext = GraphFeatureExtractor()
graph_features = graph_ext.extract_single("data/edges.csv")

merger = FeatureMerger()
merged = merger.merge_all(
    "out/query.yaml",
    "out/graph.yaml",
    "out/config.yaml",  # you author this
)
import numpy as np
X = np.array(merged["feature_matrix"])
```

---

## CostPredictor API (graph-based query)

```python
from autoconfig import CostPredictor
import networkx as nx

predictor = CostPredictor()

queries = []   # query graphs (NetworkX or dicts)
graphs = []    # data graphs
configs = []   # config dicts
times = []     # measured runtimes

metrics = predictor.train(queries, graphs, configs, times, verbose=True)

q = nx.erdos_renyi_graph(10, 0.2)
g = nx.erdos_renyi_graph(100, 0.1)
cfg = {"memory_limit": 8192, "num_threads": 4, ...}

t = predictor.predict(q, g, cfg)
pred, lo, hi = predictor.predict(q, g, cfg, return_uncertainty=True)
```

See [api_reference.md](api_reference.md) for `FeatureManager`, `BayesianExecutionTimeModel`, and related types.

---

## FAQ

**Why not instantiate symbols in step 1?** The same query can run on many graphs; decoupling keeps one query YAML reusable until merge.

**Config file?** [CONFIG_GUIDE.md](CONFIG_GUIDE.md) describes the expected YAML for merge.

**Change feature order?** Adjust `FeatureMerger` ordering (and any downstream models).

---

## See also

- [CLI_GUIDE.md](CLI_GUIDE.md)
- [feature_extraction.md](feature_extraction.md)
- [TRAIN_TEST_MERGED.md](TRAIN_TEST_MERGED.md) — `train-merged` / `eval-merged`, `--y-axis`, batch merge + filling benchmarks
- [api_reference.md](api_reference.md)
