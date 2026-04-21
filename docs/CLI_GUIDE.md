# AutoConfig CLI guide

> **Note:** `autoconfig config` currently exposes `-n`, `--k-min`, `--k-max`, and `-o` and uses the built-in default catalog. For `--resource-catalog`, `--use-default-catalog`, or `--cpu` / `--memory`, run  
> `python experiments/scripts/step3_system_config.py --help`.

## Overview

AutoConfig provides CLI tools for feature extraction and related tasks:

1. **query** — features from graph query **source code**
2. **graph** — features from edge lists (single file or partition folder)
3. **config** — configuration samples (LHS over the generator catalog)
4. **merge** — merge query + graph + config YAML into a numeric matrix
5. **all** — query + graph + config in one directory (no merge)
6. **train**, **generate-data**, **recommend** — offline / online workflows

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

## 3. Configuration generation (LHS)

### Command (`autoconfig`)

```bash
autoconfig config --num-samples <N> --output <file.yaml> [--k-min A] [--k-max B]
```

### Command (full options — pipeline script)

```bash
python experiments/scripts/step3_system_config.py \
  -n 20 -o out/config_features.yaml --use-default-catalog
```

See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for catalogs and resource YAML.

---

## 4. Merge features

```bash
autoconfig merge \
  --query out/query_features.yaml \
  --graph out/graph_features.yaml \
  --config out/config_features.yaml \
  --output out/merged_features.yaml
```

---

## 5. Full extraction (`all`)

Produces `query_features.yaml`, `graph_features.yaml`, and `config_features.yaml` in one directory (merge is separate).

```bash
autoconfig all \
  --query data/queries/kernel_bfs.cu \
  --graph data/graph_medium_pl.csv \
  --config-n 20 \
  --output out/
```

---

## 6. Training and recommendation

```bash
autoconfig train --n-samples 500 --output data/models/
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
autoconfig config -n 50 -o out/configs.yaml
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

**Custom resource catalog?** Use `step3_system_config.py` with `--resource-catalog file.yaml`.

**Why LHS?** Better coverage of the `(k, resource)` space with fewer samples than naive random grids.

**NumPy types in YAML?** `yaml.safe_load` typically returns plain Python numbers.

---

## See also

- [feature_extraction.md](feature_extraction.md) — feature definitions
- [api_reference.md](api_reference.md) — Python API
- [quickstart.md](quickstart.md) — short intro
