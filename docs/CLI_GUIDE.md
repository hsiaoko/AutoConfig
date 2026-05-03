# AutoConfig CLI guide

> **Config YAML:** the main CLI does not generate configs. Use **`data/conf/build_ten_conf.py`**
> (Latin Hypercube samples) and/or hand-authored YAML. See [CONFIG_GUIDE.md](CONFIG_GUIDE.md).

## Overview

AutoConfig provides CLI tools for feature extraction and related tasks:

1. **query** — features from graph query **source code**
2. **merge** — merge query + graph + config YAML into a numeric matrix (you supply `graph_features.yaml`)
3. **all** — write `query_features.yaml` in one directory (config / graph YAML separate)
4. **train-merged**, **eval-merged** — merged 53-D training / evaluation
5. **recommend-conf** — rank configuration candidates for one query + one graph using a trained merged model ([CONF_RECOMMEND.md](CONF_RECOMMEND.md))

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

## 2. Graph features YAML (no CLI)

This package does **not** include a `graph` subcommand. Author **`graph_features.yaml`** yourself (or use an external tool) to match the format expected by **merge**; see [feature_extraction.md](feature_extraction.md).

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

## 4. Query-only batch (`all`)

Writes **`query_features.yaml`** under the output directory. Requires **`--query`**. Add `graph_features.yaml` and `config_features.yaml` yourself, then run **merge**.

```bash
autoconfig all \
  --query data/queries/kernel_bfs.cu \
  --output out/
```

---

## 5. Training

**Merged YAML trainer / evaluator** (53-D: **X** = features 4–53; **Y** = **`-y` / `--y-axis`** with `0|1|2` — see [TRAIN_TEST_MERGED.md](TRAIN_TEST_MERGED.md#label-y-in-the-cli--y----y-axis)). Full detail: [TRAIN_TEST_MERGED.md](TRAIN_TEST_MERGED.md) (including `merge_abc_features.py` → `fill_merged_train_costs_from_stats.py` → `run_train_merged.sh`).

**Model backend:** default is **`--model-kind bayesian`**. For a neural net use **`--model-kind nn`** (or **`mlp`**) and optional **`--model-options '{"max_iter":500,...}'`**. **`--model-kind rl`** is a stub and does not train until you register a real implementation — see [TRAIN_TEST_MERGED.md — Model backend](TRAIN_TEST_MERGED.md#model-kind).

```bash
# Train (default Y = time = -y 1); writes .pkl + _meta.yaml
autoconfig train-merged --data-dir out/train --output out/models

# Train with a neural network (same data layout)
autoconfig train-merged --data-dir out/train --output out/models --model-kind nn

# Train with Y = price (y = 0)
autoconfig train-merged -d out/train -o out/models -y 0

# Evaluate on a test folder; prints MAE, RMSE, MAPE, R²
autoconfig eval-merged -m out/models/bayesian_cost_merged.pkl -d out/test

# Train via helper (must pass --model-basename <stem>)
./scripts/run_train_merged.sh --model-basename bayesian_cost_full -d out/train -o out/models
```

For **merged 53-D** models, use **MergedBayesianPredictor** in `autoconfig.merged` (see [api_reference.md](api_reference.md)). For raw code + graph + config features (no merged YAML), optionally use `autoconfig.prediction.CostPredictor`.

---

## 6. Recommend configuration (`recommend-conf`)

After **`train-merged`**, rank **many** candidate configs for **one** task (`-q`) and **one** graph (`-g`) by predicted **`price` / `time` / `cost`** (whichever the model was trained on — read from `*_meta.yaml`).

```bash
autoconfig recommend-conf \
  -m out/models/my_run.pkl \
  -q out/query_features/kernel_bfs.yaml \
  -g out/graph_features/friendster.yaml \
  -c data/conf/gpu/ \
  -k 5 \
  -o out/recommended_confs

# Wrapper (same CLI):
./scripts/run_recommend_conf.sh -m out/models/my_run.pkl -q Q -g G -c data/conf/gpu/ -k 5 -o out/rec
```

**Full behavior** (config directory vs single file, `-o` directory vs summary YAML, `--refine`, Python API): **[CONF_RECOMMEND.md](CONF_RECOMMEND.md)**.

---

## Usage scenarios

### Prepare training tensors

```bash
for f in data/queries/*.cu; do
  autoconfig query -i "$f" -o "out/queries/$(basename "$f" .cu).yaml"
done
# provide out/graph_features.yaml (not generated by autoconfig CLI)
# e.g. python data/conf/build_ten_conf.py -n 10  (see CONFIG_GUIDE.md) before merge
```

### Inspect query workload patterns

Open the query YAML and inspect symbolic / static groups (or use `FeatureMerger` after graph features exist).

### Partition quality

Put partition statistics into **`graph_features.yaml`** (see [feature_extraction.md](feature_extraction.md)) so fields like `edge_cut_ratio` and `balance` are available to **merge**.

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
