# AutoConfig quick start

## Five-minute tour

### Install

```bash
cd AutoConfig
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Full pipeline example

```bash
bash examples/run_full_pipeline.sh
```

This typically:

1. Extracts query code features (C++/CUDA)
2. Extracts graph features
3. Generates capacity-aware configurations (if the script uses capacity YAML)
4. Merges into a 47-dimensional feature vector

### Step by step

#### 1. Query features

```bash
autoconfig query --input data/queries/kernel_bfs.cu -o out/query.yaml
```

**Output**: 20 features (8 static + 12 symbolic placeholders or template fields, depending on format).

#### 2. Graph features

```bash
autoconfig graph --input data/graph_medium_pl.csv -o out/graph.yaml
```

**Output**: graph statistics (vertices, edges, degree distribution, structure, partition fields when applicable).

#### 3. Configurations

```bash
# CLI entry (built-in default catalog)
autoconfig config -n 5 -o out/config.yaml

# Pipeline script: default catalog, custom catalog, or CPU/memory simple catalog
python experiments/scripts/step3_system_config.py -n 5 -o out/config.yaml --use-default-catalog

python experiments/scripts/step3_system_config.py -n 5 -o out/config.yaml \
  --cpu 16 --memory 64
```

System **capacity** YAMLs under `data/conf/` are documented in [docs/CONFIG_GUIDE.md](docs/CONFIG_GUIDE.md) for understanding cluster limits when you extend the generator.

#### 4. Merge

```bash
autoconfig merge \
    --query out/query.yaml \
    --graph out/graph.yaml \
    --config out/config.yaml \
    -o out/merged.yaml
```

**Output**: feature matrix whose width matches `FeatureMerger` (commonly 47 features × _n_ config samples).

---

## Data files

### Query examples

- `data/queries/kernel_*.cu` — CUDA kernels for feature extraction tests

### Sample graphs

| File pattern | Nodes | Edges (approx.) | Type |
|--------------|-------|-----------------|------|
| `data/graph_small_*.csv` | 100 | ~250 | Small |
| `data/graph_medium_*.csv` | 1,000 | ~5,000 | Medium |
| `data/graph_large_*.csv` | 5,000 | ~50,000 | Large |

### System capacity YAML

| File | CPU | Memory | GPU | Max machines |
|------|-----|--------|-----|--------------|
| `data/conf/system_capacity_small.yaml` | 64C | 256GB | 4 | 4 |
| `data/conf/system_capacity_medium.yaml` | 256C | 1TB | 16 | 8 |
| `data/conf/system_capacity_large.yaml` | 512C | 2TB | 32 | 8 |
| `data/conf/system_capacity_cpu_only.yaml` | 128C | 512GB | 0 | 4 |

---

## CLI tools

### `query` — query code features

```bash
autoconfig query --input <source file> -o <output.yaml>
```

- `--input`, `-i`: query source (required)
- `--output`, `-o`: output path (default: `out/query_features.yaml`)

### `graph` — graph features

```bash
autoconfig graph --input <edge CSV or directory> -o <output.yaml>
```

- `--input`, `-i`: file or partitioned graph folder (required)
- `--output`, `-o`: output path (default: `out/graph_features.yaml`)

### `config` — sample configurations

```bash
autoconfig config -n <samples> -o <output.yaml> [--k-min N] [--k-max M]
```

For catalogs, capacity, and `--cpu` / `--memory`, see [docs/CONFIG_GUIDE.md](docs/CONFIG_GUIDE.md) and `experiments/scripts/step3_system_config.py`.

### `merge` — merge YAMLs

```bash
autoconfig merge -q <query.yaml> -g <graph.yaml> -c <config.yaml> -o <merged.yaml>
```

---

## Output sketches

### Query YAML

```yaml
query_features:
  static:
    static_loop_count: 8.0
    static_atomic_op_count: 4.0
    static_sync_count: 7.0
  symbolic:
    sym_vscan_coeff: 1.0
    ...
```

### Merged YAML

```yaml
feature_matrix:
  - [8.0, 2.0, ..., 64.0]
metadata:
  num_samples: 5
  num_features: 47
```

---

## Feature groups (typical merged layout)

| Group | Count | Role |
|-------|-------|------|
| Static | 8 | Query structure |
| Symbolic | 12 | Workload templates (instantiated at merge) |
| Graph | 17 | Graph / partition stats |
| Config | 10 | Resource configuration |
| **Total** | **47** | |

---

## Next steps

- [docs/CLI_GUIDE.md](docs/CLI_GUIDE.md) — detailed CLI
- [docs/CONFIG_GUIDE.md](docs/CONFIG_GUIDE.md) — configuration generation
- [docs/feature_extraction.md](docs/feature_extraction.md) — feature definitions
- [docs/usage_guide.md](docs/usage_guide.md) — full workflow and Python API