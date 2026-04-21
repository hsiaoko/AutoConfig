# Recommendation updates

## 1. Query **files** instead of enum names

**Before**

```bash
autoconfig recommend --query bfs --graph data/graph.csv
```

Only built-in query names (`bfs`, `dfs`, `pagerank`, …) were supported.

**After**

```bash
autoconfig recommend --query my_query.cu --graph data/graph.csv
```

Any `.cu`, `.cpp`, or `.py` file can be analyzed for complexity-style signals.

## 2. Diverse Top-3 (no duplicate configs)

**Before:** repeated ranks with identical resources.

**After:** three **distinct** resource signatures, sorted by predicted cost.

## New component: `QueryComplexityExtractor`

**Path:** `autoconfig/utils/query_complexity_extractor.py`

Heuristic patterns include:

- **v_scan** — vertex-oriented loops (`forAllVertices`, `parallel_for(0, numVertices)`, …)  
- **e_scan** — edge-oriented loops  
- **f_scan** — frontier / neighbor iteration  
- **atomic** — `atomicAdd`, `fetch_add`, …  
- **sync** — barriers / `__syncthreads` / `MPI_Barrier`, …  

```python
from autoconfig.utils.query_complexity_extractor import QueryComplexityExtractor

extractor = QueryComplexityExtractor()
complexity = extractor.extract_from_file("my_query.cu")
# e.g. {'v_scan': 1, 'e_scan': 0, 'f_scan': 1, 'atomic': 1, 'sync': 0}
```

## Code changes (summary)

- **`cli.py`** — `--query` accepts a file path; complexity is extracted before recommendation.  
- **`optimizer.py`** — configuration signatures, perturbations, and diverse top-K selection.  
- **`recommender.py`** — `recommend_with_complexity(query_complexity, graph_features, ...)`.

## Usage

```bash
autoconfig recommend \
    --query my_algorithm.cu \
    --graph data/edges.csv \
    --top-n 3 \
    --output out/recommendation.yaml
```

```python
from autoconfig.online.recommender import Recommender
from autoconfig.utils.query_complexity_extractor import QueryComplexityExtractor

extractor = QueryComplexityExtractor()
query_complexity = extractor.extract_from_file("my_query.cu")

recommender = Recommender(model_dir="data/models/")
result = recommender.recommend_with_complexity(
    query_complexity=query_complexity,
    graph_features={
        "num_vertices": 10000,
        "num_edges": 50000,
        "avg_degree": 5.0,
        "max_degree": 100,
        "density": 0.001,
        "avg_clustering": 0.5,
    },
    top_n=3,
)
```

## Deduplication

Resource signatures distinguish recommendations, e.g. `CPU:…_MEM:…_GPU:…`.

## Backward compatibility

`recommend(query_name='bfs', ...)` remains available and routes through the same stack where applicable.
