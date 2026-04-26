# AutoConfig CLI guide

> **Config YAML:** the main CLI does not generate configs. Use **`data/conf/build_ten_conf.py`**
> (Latin Hypercube samples) and/or hand-authored YAML. See [CONFIG_GUIDE.md](CONFIG_GUIDE.md).

## Overview

AutoConfig provides CLI tools for feature extraction and related tasks:

1. **query** — features from graph query **source code**
2. **graph** — features from edge lists (single file or partition folder)
3. **merge** — merge query + graph + config YAML into a numeric matrix
4. **all** — query + graph in one directory (config YAML is separate; no merge)
5. **train**, **train-merged**, **eval-merged**, **generate-data**, **recommend** — offline / online workflows

YAML outputs are usually written under `out/`.

---

## Install

```bash
cd AutoConfig
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## 1. Query code features

Extract **static** and **symbolic** features from query source.

### Command

```bash
autoconfig query --input <file> --output <output.yaml>
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input`, `-i` | Query source file (required) | — |
| `--output`, `-o` | Output YAML path | `out/query_features.yaml` |

### Examples

```bash
autoconfig query --input data/queries/kernel_bfs.cu --output out/query_features.yaml
```

### Sample output

```yaml
feature_count:
  static: 8
  symbolic: 12
  total: 20
metadata:
  input_file: data/queries/kernel_bfs.cu
query_features:
  static:
    static_loop_count: 2.0
    ...
  symbolic:
    ...
```

### Language patterns

- **Python**: `for v in G.vertices():`, `while cond:`
- **C/C++**: `for(;;)`, `while()`, `if()`
- **Pseudocode-style**: `for v in Vertices:`, `if !condition:`

---

## 2. Graph features

From edge-list CSV or a **directory of shards**.

### Command

```bash
autoconfig graph --input <file_or_folder> --output <output.yaml>
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input`, `-i` | Edge CSV or partition directory (required) | — |
| `--output`, `-o` | Output YAML path | `out/graph_features.yaml` |

### Edge list CSV

```csv
src,dst
0,1
0,2
1,2
```

The first row may be a header (`src,dst`) or raw edges only.

### Examples

```bash
autoconfig graph --input data/graph_medium_pl.csv --output out/graph_features.yaml
autoconfig graph --input data/graph_small_partitioned/ --output out/graph_features.yaml
```

Partition folders should contain multiple edge files (`*.csv`, `*.edges`, `*.txt`).

---

## 3. Merge features

```bash
autoconfig merge \
  --query out/query_features.yaml \
  --graph out/graph_features.yaml \
  --config out/config_features.yaml \
  --output out/merged_features.yaml
```

---

## 4. Full extraction (`all`)

Writes `query_features.yaml` and `graph_features.yaml` under the output directory. Create `config_features.yaml` yourself, then run **merge** separately.

```bash
autoconfig all \
  --query data/queries/kernel_bfs.cu \
  --graph data/graph_medium_pl.csv \
  --output out/
```

---

## 5. Training and recommendation

**Synthetic-data trainer** (legacy internal feature format):

```bash
autoconfig train --n-samples 500 --output data/models/
```

**Merged YAML trainer / evaluator** (1× `cost` + 50 inputs per file; same layout as `merge` output). Full detail: [TRAIN_TEST_MERGED.md](TRAIN_TEST_MERGED.md) (including `scripts/merge_abc_features.py` → `fill_merged_train_costs_from_stats.py` → `run_train_merged.sh`).

```bash
# Train; writes e.g. out/models/bayesian_cost_merged.pkl + _meta.yaml
autoconfig train-merged --data-dir out/train --output out/models

# Evaluate on a test folder; prints MAE, RMSE, MAPE, R²
autoconfig eval-merged -m out/models/bayesian_cost_merged.pkl -d out/test

# Optional: helper scripts
# ./scripts/run_train_merged.sh
# ./scripts/run_eval_merged.sh -m out/models/bayesian_cost_merged.pkl -d out/test
```

**Online recommendation (separate model stack in code):**

```bash
autoconfig recommend --query my_kernel.cu --graph data/graph.csv --top-n 3
```

---

## Usage scenarios

### Prepare training tensors

```bash
for f in data/queries/*.cu; do
  autoconfig query -i "$f" -o "out/queries/$(basename "$f" .cu).yaml"
done
autoconfig graph -i data/graph_medium_pl.csv -o out/graph.yaml
# e.g. python data/conf/build_ten_conf.py -n 10  (see CONFIG_GUIDE.md) before merge
```

### Inspect query workload patterns

Open the query YAML and inspect symbolic / static groups (or use `FeatureMerger` after graph features exist).

### Partition quality

```bash
autoconfig graph -i data/partitions/ -o out/partitions.yaml
```

Use `edge_cut_ratio`, `balance`, and related fields in the YAML.

---

## YAML layout (reference)

### Query

```yaml
feature_count: { static: 8, symbolic: 12, total: 20 }
query_features:
  static: { ... }
  symbolic: { ... }
```

### Graph

```yaml
graph_features:
  basic: { ... }
  degree: { ... }
  structure: { ... }
  partition: { ... }
  quality_metrics: { ... }   # partitioned input only
partition_info: { ... }      # partitioned input only
```

### Config

```yaml
configurations: [ ... ]
config_features: [ ... ]
metadata: { ... }
```

---

## FAQ

**Invalid edge list?** Use `src,dst` with one edge per line, or omit the header.

**Config candidates?** [CONFIG_GUIDE.md](CONFIG_GUIDE.md) and `data/conf/build_ten_conf.py` (LHS).

**NumPy types in YAML?** `yaml.safe_load` typically returns plain Python numbers.

---

## See also

- [TRAIN_TEST_MERGED.md](TRAIN_TEST_MERGED.md) — merged-feature train / test
- [feature_extraction.md](feature_extraction.md) — feature definitions
- [api_reference.md](api_reference.md) — Python API
- [quickstart.md](quickstart.md) — short intro
