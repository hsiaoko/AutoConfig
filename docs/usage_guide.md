# AutoConfig usage guide

## Overview

End-to-end feature workflow:

1. **Query** — static + symbolic templates from source
2. **Graph** — structure and partition statistics
3. **Config** — LHS samples over a resource catalog
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

```bash
autoconfig config --num-samples 20 --output out/config_features.yaml
```

For catalogs and `--cpu` / `--memory`, use `experiments/scripts/step3_system_config.py` ([CONFIG_GUIDE.md](CONFIG_GUIDE.md)).

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
python experiments/scripts/step3_system_config.py -n 3 -o out/config.yaml --use-default-catalog
autoconfig merge -q out/query.yaml -g out/graph.yaml -c out/config.yaml -o out/merged.yaml
```

---

## Feature groups (typical 47-D merge)

| Group | Count | Role |
|-------|-------|------|
| Static | 8 | Query structure |
| Symbolic | 12 | Templates → numeric at merge |
| Graph | 17 | Graph / partition |
| Config | 10 | Resources |
| **Total** | **47** | |

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

## Python API (extractors + merger)

```python
from autoconfig.utils.query_feature_extractor import QueryFeatureExtractor
from autoconfig.utils.graph_feature_extractor import GraphFeatureExtractor
from autoconfig.utils.config_generator import ConfigGenerator, generate_default_catalog
from autoconfig.utils.feature_merger import FeatureMerger  # merge_all loads YAML paths

query_ext = QueryFeatureExtractor()
query_features = query_ext.extract_from_file("data/queries/kernel_bfs.cu")

graph_ext = GraphFeatureExtractor()
graph_features = graph_ext.extract_single("data/edges.csv")

catalog = generate_default_catalog()
gen = ConfigGenerator(catalog)
config_bundle = gen.generate(20, (1, 16))

merger = FeatureMerger()
merged = merger.merge_all(
    "out/query.yaml",
    "out/graph.yaml",
    "out/config.yaml",
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

**Custom catalogs?** Pass `--resource-catalog` to `step3_system_config.py`.

**Change feature order?** Adjust `FeatureMerger` ordering (and any downstream models).

---

## See also

- [CLI_GUIDE.md](CLI_GUIDE.md)
- [feature_extraction.md](feature_extraction.md)
- [api_reference.md](api_reference.md)
