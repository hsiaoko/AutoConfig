# Feature extraction refactor — summary

The feature stack was rebuilt around the paper’s multi-stage design: **static**, **symbolic templates**, **graph/partition**, and **config** features, then merged into a single numeric vector.

## 1. Static extractor (`static_extractor.py`)

Eight structural features, e.g. loop count, max loop depth, branches, variables, recursion, atomics, sync primitives, explicit parallelism.

Supported idioms: C/C++ loops and conditionals, Python `for` / `while`, pseudocode-style patterns.

Classes: `StaticFeatureExtractor` (regex-oriented), `ASTBasedStaticExtractor` (Python AST).

## 2. Symbolic extractor (`symbolic_extractor.py`)

Six templates → twelve scalars (`coeff` + `requires` pairs): **VScan**, **EScan**, **FScan**, **RExp**, **Atom**, **Comm**, with formulas documented in `feature_extraction.md`.

## 3. Graph / partition extractor (`graph_partition_extractor.py`)

Seventeen graph and partition statistics (vertices, edges, degrees, diameter, clustering, cut, balance, …).

Helper: `PartitionQualityMetrics`.

## 4. Feature manager (`feature_manager.py`)

Orchestrates the pipeline; typical merged width **47** (8 + 12 + 17 + 10).

## 5. Config extractor (`config_extractor.py`)

Ten configuration scalars (memory, threads, cache, batching, I/O, indexing, compression, timeout, …).

## Files touched

New / updated under `autoconfig/feature_extractor/`, examples, and `docs/feature_extraction.md`.

## Example

```python
from autoconfig import FeatureManager
import networkx as nx

manager = FeatureManager()
code = """
for v in G.vertices():
    for neighbor in G.neighbors(v):
        process(v, neighbor)
"""
graph = nx.erdos_renyi_graph(1000, 0.05)
config = {"memory_limit": 8192, "num_threads": 4}
features = manager.extract_all(code, graph, config)
```

## Tests

Run `PYTHONPATH=. python tests/test_autoconfig.py -v` (see project state for exact count).

## Documentation

| File | Role |
|------|------|
| `docs/feature_extraction.md` | Feature definitions |
| `README.md` | Project overview |
| `docs/usage_guide.md` | Usage and API entry points |
| `docs/api_reference.md` | Class-level reference |

## Compatibility notes

Older APIs that expected a **query graph** as the first argument to `FeatureManager.extract_all` may need to pass **query source code** (string) instead. Migrate from `extract_all(query_graph, …)` to `extract_all(query_code_string, …)`.

## Roadmap ideas

- LLM-assisted symbolic detection  
- Richer pattern libraries  
- Incremental features on dynamic graphs  
- Automated feature selection  

## Paper cross-references

Refer to your paper’s ML feature section and training subsection for the full theoretical context.
