# Feature Extraction Methods

This document describes the three-stage feature extraction pipeline for graph query execution time prediction.

## Overview

Our approach extracts features from three sources:

1. **Query Code** - Static and symbolic features from the query program
2. **Graph Data** - Structural and partition statistics from the input graph
3. **System Configuration** - Resource allocation parameters

The feature extraction is based on the method described in the research paper for Hybrid graph query tasks.

---

## 1. Query Code Feature Extraction

### Static Features (8 features)

Extracted from the query source code through lightweight static analysis. These features capture the program structure independent of the input graph.

| # | Feature | Code Pattern | Performance Relevance |
|---|---------|--------------|----------------------|
| 1 | `static_loop_count` | Loop constructs (for/while) | Repeated work regions |
| 2 | `static_max_loop_depth` | Nested loops | Nested work growth |
| 3 | `static_branch_count` | Conditional branches (if/switch) | Control-flow irregularity |
| 4 | `static_variable_count` | Variable definitions | Local state size |
| 5 | `static_recursion_count` | Recursive procedures | Search and propagation expansion |
| 6 | `static_atomic_op_count` | Atomic updates on shared state | Parallel contention risk |
| 7 | `static_sync_count` | Locks, barriers, synchronization | Coordination overhead |
| 8 | `static_explicit_parallel_flag` | PRAM operator invocations | Parallel-execution overheads |

**Example Output:**
```yaml
query_features:
  static:
    static_loop_count: 2.0
    static_max_loop_depth: 1.0
    static_branch_count: 3.0
    static_variable_count: 7.0
    static_recursion_count: 1.0
    static_atomic_op_count: 0.0
    static_sync_count: 1.0
    static_explicit_parallel_flag: 0.0
```

### Symbolic Features (12 features)

Symbolic workload templates that capture dominant computation, communication, and synchronization patterns. These are partial functions instantiated with graph/partition statistics.

| # | Template | Code Pattern | Instantiation | Requires |
|---|----------|--------------|---------------|----------|
| 1 | **VScan**(V, n) | Vertex-list scans, PRAM vertex operators | \|V\| or \|V\|/n | \|V\| |
| 2 | **EScan**(E, n) | Edge-list scans, neighbor traversals | \|E\| or \|E\|/n | \|E\| |
| 3 | **FScan**(G, n) | Worklist/frontier-driven loops | ∑_{t=1}^{D} \|E_t\| | D (diameter) |
| 4 | **RExp**(G) | Recursive neighbor expansion | ∏_{i=1}^{D} E[deg(v_i)] | D, avg_degree |
| 5 | **Atom**(G) | Atomic writes, compare-and-swap | \|E\| × skew(G) | \|E\|, skew |
| 6 | **Comm**(G, F) | Cross-partition communication | ∑_{v ∈ V_∂} deg_∂(v) | boundary-degree stats |

**Pattern Recognition:**
- Vertex scanning: `for v in G.vertices()`, `Vertices()`
- Edge scanning: `for neighbor in G.neighbors(v)`, `out_edges()`
- Frontier iteration: `while !worklist.empty()`, `frontier.next()`
- Recursive expansion: `Expand(m, level+1, ...)`, recursive DFS/BFS
- Atomic updates: `atomicAdd()`, `compare_and_swap()`
- Cross-partition: `SendTo()`, `IsMirror()`, `Owner()`

**Example Output:**
```yaml
query_features:
  symbolic:
    sym_vscan_coeff: 0.0
    sym_vscan_requires: 0.0
    sym_escan_coeff: 50000.0      # |E| edges
    sym_escan_requires: 1.0
    sym_fscan_coeff: 50000.0      # D × |E|/D
    sym_fscan_requires: 1.0
    sym_rexp_coeff: 0.0
    sym_rexp_requires: 0.0
    sym_atom_coeff: 0.0
    sym_atom_requires: 0.0
    sym_comm_coeff: 0.0
    sym_comm_requires: 0.0
```

---

## 2. Graph Data Feature Extraction

### Basic Statistics (2 features)

| Feature | Description |
|---------|-------------|
| `graph_num_vertices` | Number of vertices \|V\| |
| `graph_num_edges` | Number of edges \|E\| |

### Degree Statistics (5 features)

| Feature | Description |
|---------|-------------|
| `graph_avg_degree` | Average degree (2\|E\|/\|V\|) |
| `graph_max_degree` | Maximum degree |
| `graph_min_degree` | Minimum degree |
| `graph_degree_std` | Degree standard deviation |
| `graph_skew` | Degree skew (max_degree / avg_degree) |

### Structural Statistics (3 features)

| Feature | Description |
|---------|-------------|
| `graph_diameter` | Graph diameter (longest shortest path) |
| `graph_clustering_coeff` | Average clustering coefficient |
| `graph_num_components` | Number of connected components |

### Partition Statistics (7 features)

For partitioned graphs, we extract:

| Feature | Description |
|---------|-------------|
| `partition_num_partitions` | Number of partitions |
| `partition_boundary_vertices` | Vertices with neighbors in other partitions |
| `partition_boundary_degree_sum` | Sum of boundary vertex degrees |
| `partition_avg_partition_size` | Average vertices per partition |
| `partition_size_std` | Partition size standard deviation |
| `partition_edge_cut_ratio` | Fraction of edges crossing partitions |
| `partition_balance` | Balance score (1.0 = perfectly balanced) |

### Quality Metrics (4 features, for partitioned graphs)

| Feature | Description |
|---------|-------------|
| `edge_cut_ratio` | Edge cut ratio |
| `balance_score` | Partition balance (avg_size / max_size) |
| `boundary_ratio` | Fraction of boundary vertices |
| `comprehensive_score` | Weighted combination: 0.5×cut + 0.3×imbalance + 0.2×boundary |

**Example Output (Single Graph):**
```yaml
graph_features:
  basic:
    num_vertices: 10000
    num_edges: 50000
    density: 0.001
  degree:
    avg: 10.0
    max: 150.0
    min: 2.0
    std: 8.5
    skew: 15.0
  structure:
    diameter: 6
    clustering_coeff: 0.05
    num_components: 1
  partition:
    num_partitions: 1
    boundary_vertices: 0
    edge_cut_ratio: 0.0
    balance: 1.0
```

**Example Output (Partitioned Graph):**
```yaml
graph_features:
  basic:
    num_vertices: 10000
    num_edges: 50000
  partition:
    num_partitions: 4
    boundary_vertices: 1500
    boundary_degree_sum: 4500.0
    avg_partition_size: 2500.0
    partition_size_std: 100.5
    edge_cut_ratio: 0.25
    balance: 0.95
  quality_metrics:
    edge_cut_ratio: 0.25
    balance_score: 0.95
    boundary_ratio: 0.15
    comprehensive_score: 0.22
partition_info:
  '0':
    file: partitions/partition_0.csv
    num_edges: 12000
    num_vertices: 2450
  '1':
    file: partitions/partition_1.csv
    num_edges: 13000
    num_vertices: 2550
  ...
```

---

## 3. Configuration Feature Extraction

### Configuration Model

A configuration is a tuple `C = (k, s)` where:
- `k`: Number of instances
- `s ∈ S`: Resource description from catalog `S = {s_1, ..., s_m}`

Each resource `s_i` specifies:
- `cpu_cores`: CPU core count
- `memory_gb`: DRAM capacity (GB)
- `storage_gb`: SSD/storage capacity (GB)
- `num_gpus`: Number of GPUs
- `gpu_sm_count`: Streaming multiprocessors per GPU (optional)
- `gpu_memory_gb`: GPU memory capacity (GB, optional)

### Configuration Features (10 features)

| Feature | Description |
|---------|-------------|
| `conf_k_instances` | Number of instances (k) |
| `conf_total_cpu_cores` | k × cpu_cores |
| `conf_total_memory_gb` | k × memory_gb |
| `conf_total_storage_gb` | k × storage_gb |
| `conf_total_gpus` | k × num_gpus |
| `conf_per_instance.cpu_cores` | CPU cores per instance |
| `conf_per_instance.memory_gb` | Memory per instance |
| `conf_per_instance.storage_gb` | Storage per instance |
| `conf_per_instance.num_gpus` | GPUs per instance |
| `conf_resource_type` | Resource type ID |

### Latin Hypercube Sampling (LHS)

We use LHS to generate a compact candidate set Σ_C from the resource catalog:

1. **Sample Generation**: Generate LHS samples in [0,1]^d where d = number of resource dimensions
2. **Resource Matching**: Find closest resource in catalog for each sample
3. **Instance Sampling**: Sample k from [k_min, k_max] based on LHS
4. **Deduplication**: Remove duplicate (k, resource) pairs

**Advantages:**
- Broad coverage of configuration space with limited samples
- Each dimension is stratified equally
- More efficient than random sampling

**Example Output:**
```yaml
configurations:
  - config_id: 0
    k: 4
    resource:
      cpu_cores: 16
      memory_gb: 64
      storage_gb: 500
      num_gpus: 1
      gpu_memory_gb: 16
  - config_id: 1
    k: 8
    resource:
      cpu_cores: 32
      memory_gb: 128
      storage_gb: 1000
      num_gpus: 4
      gpu_memory_gb: 32
  ...
config_features:
  - conf_k_instances: 4
    conf_total_cpu_cores: 64
    conf_total_memory_gb: 256
    conf_total_storage_gb: 2000
    conf_total_gpus: 4
    conf_per_instance:
      cpu_cores: 16
      memory_gb: 64
      storage_gb: 500
      num_gpus: 1
metadata:
  num_samples: 20
  k_range: [1, 16]
  catalog_size: 11
  sampling_method: latin_hypercube
catalog_stats:
  cpu_cores:
    min: 2
    max: 64
    mean: 18.5
    std: 15.2
  memory_gb:
    min: 4
    max: 256
    mean: 68.2
    std: 58.3
```

---

## Combined Feature Vector

The complete feature vector combines all features:

```
Φ(Q, G, C) = [Φ_static(Q), Φ_sym(Q, G), Φ_graph(G), Φ_config(C)]
```

**Dimensions:**
- Static: 8
- Symbolic: 12
- Graph/Partition: 17
- Configuration: 10
- **Total: 47**

---

## Usage Examples

### Extract Query Features

```bash
autoconfig query \
    --input queries/bfs.py \
    --output out/query_features.yaml \
    --num-vertices 10000 \
    --num-edges 50000 \
    --diameter 6
```

### Extract Graph Features

```bash
# Single graph
autoconfig graph \
    --input data/twitter_edges.csv \
    --output out/graph_features.yaml

# Partitioned graph
autoconfig graph \
    --input data/partitions/ \
    --output out/graph_features.yaml
```

### Generate Configurations

```bash
autoconfig config \
    --num-samples 20 \
    --output out/config_features.yaml \
    --use-default-catalog \
    --k-min 1 \
    --k-max 16
```

### Complete Pipeline

```bash
autoconfig all \
    --query queries/bfs.py \
    --graph data/twitter_edges.csv \
    --config-n 20 \
    --output out/
```

---

## Feature Importance Analysis

After training the model, feature importance can be analyzed:

```python
from autoconfig import CostPredictor

predictor = CostPredictor()
predictor.train(queries, graphs, configs, times)

# Get feature importance
importance = predictor.get_feature_importance()
names = predictor.get_feature_names()

# Display top features
for i in importance.argsort()[::-1][:10]:
    print(f"{names[i]}: {importance[i]:.4f}")
```

**Typical Important Features:**
- `sym_escan_coeff` - Edge scanning workload
- `graph_num_edges` - Graph size
- `conf_total_cpu_cores` - Computational resources
- `sym_fscan_coeff` - Frontier iteration workload
- `partition_edge_cut_ratio` - Communication overhead

---

## Performance Characteristics

### Feature Extraction Speed

| Stage | Time per Sample |
|-------|-----------------|
| Static features | ~0.1 ms |
| Symbolic features | ~0.2 ms |
| Graph features (1K nodes) | ~10 ms |
| Configuration features | ~0.1 ms |
| **Total** | **< 15 ms** |

### Scalability

- **Static/Symbolic**: O(|code|) - linear in code size
- **Graph**: O(|V| + |E|) - linear in graph size (with sampling for large graphs)
- **Configuration**: O(1) - constant time

---

## References

1. **Static and Symbolic Features**: Table 1 in the research paper
2. **Latin Hypercube Sampling**: McKay, M. D., Beckman, R. J., & Conover, W. J. (1979)
3. **Bayesian Model Training**: See `docs/api_reference.md`

---

## Appendix: Feature Templates

### Vertex Scanning (VScan)
```
Pattern: for v in G.vertices()
Cost: |V| / n  (where n = parallelism factor)
```

### Edge Scanning (EScan)
```
Pattern: for neighbor in G.neighbors(v)
Cost: |E| / n
```

### Frontier Iteration (FScan)
```
Pattern: while !worklist.empty()
Cost: Σ_{t=1}^{D} |E_t| ≈ D × |E| / D
```

### Recursive Expansion (RExp)
```
Pattern: Expand(m, level+1, ...)
Cost: ∏_{i=1}^{D} E[deg(v_i)] ≈ avg_degree^D
```

### Atomic Updates (Atom)
```
Pattern: atomicAdd(counter[v], value)
Cost: |E| × skew(G)  (where skew = max_degree / avg_degree)
```

### Cross-Partition Communication (Comm)
```
Pattern: SendTo(owner(v), message)
Cost: Σ_{v ∈ V_∂} deg_∂(v)
```
