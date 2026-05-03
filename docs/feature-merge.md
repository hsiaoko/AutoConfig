# Feature merge

Combine **query**, **graph**, and **config** YAML into one (or a multi-row matrix) **53-D** vector: `price`, `time`, `cost` + `static_*` + `sym_*` + graph/partition scalars + `conf_*`. Training uses the last **50** dimensions as **X** (indices 3…52); the first three are optional **Y** (`-y 0|1|2`).

## CLI: `autoconfig merge`

```bash
autoconfig merge -h
```

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--query` | `-q` | yes | — | Query-feature YAML. |
| `--graph` | `-g` | yes | — | Graph-feature YAML. |
| `--config` | `-c` | yes | — | Merge-format config YAML. |
| `--output` | `-o` | no | `out/merged_features.yaml` | Merged output YAML. |

If `config` lists multiple `configurations`, `feature_matrix` may have one row per configuration.

## Batch script: `scripts/merge_abc_features.py`

Cartesian product of **config dir × graph dir × query dir**; one output YAML per triple: `{config_stem}_{graph_stem}_{query_stem}.yaml`.

```bash
python scripts/merge_abc_features.py -h
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--query-dir` | yes | — | Directory of query YAMLs. |
| `--graph-dir` | yes | — | Directory of graph YAMLs. |
| `--config-dir` | yes | — | Directory of config YAMLs. |
| `--out-dir` | yes | — | Output directory. |
| `--config-glob` | no | `conf_*.yaml` | Glob for config files. |
| `--query-glob` | no | `*.yaml` | Glob under `--query-dir`. |
| `--query-stems` | no | `None` | If set: keep only query files whose `Path.stem` is in this list (space-separated `STEM` values). |
| `--graph-recursive` | no | off (flag) | If set: include `**/*.yaml` under `--graph-dir`. |
| `-v`, `--verbose` | no | off | Print each path; stderr note when a config file has multiple rows but only the first is used. |

**Note:** If a config file has several `configuration` entries, this script uses **only the first** so the file count stays exactly A×B×C.

## Optional: fill `time` / `cost` for training

After merge, `time` and `cost` are often still placeholders; you can backfill from benchmark stats:

### `scripts/fill_merged_train_costs_from_stats.py`

Reads GridGraph `STATS.txt` (or `conf_*.output` / `real` lines if `STATS.txt` is missing); sets `time` and `cost = price × time` unless `--no-cost-formula`.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--no-cost-formula` | flag | off | Update `time` only; set `cost` to 0. |
| `--gridgraph-root` | Path | parent repo’s `GridGraph` | Root containing `*_gridgraph_*/STATS.txt`. |
| `--features-dir` | Path | `<repo>/out/features` | Input merged `conf_*_*_gridgraph_*.yaml` style files. |
| `--output-dir` | Path | `<repo>/out/train` | Output (can equal input for in-place update). |
| `--no-validate-cpu-mem` | flag | off | Do not require STATS CPU/memory to match YAML `config_scalars`. |
| `--strict` | flag | off | Exit on missing table, empty parse, or mismatch (when validation on). |

### `scripts/fill_merged_time_cost_from_gridgraph_stats.py`

Uses dataset / GridGraph task path layout; updates selected `query_stem` values (`gridgraph_sssp`, `gridgraph_coloring`).

| Argument | Required | Description |
|----------|----------|-------------|
| `--merged-dir` | yes | Directory of merged YAMLs. |
| `--gridgraph-exp` | yes | GridGraph `exp` root. |
| `--dry-run` | no | Statistics only, no writes. |

### `scripts/fill_merged_time_cost_matrixgraph_gpu.py`

Aligns MatrixGraph `exp/processed_gpu/.../STATS.txt` with `query_stem`, `graph_stem`, and `config_stem` (`conf_NN`); writes `time` / `cost`.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--merged-dir` | Path | **required** | e.g. `exp/features/gpu`. |
| `--matrixgraph-processed` | Path | `/home/zhuxk/project/graph/MatrixGraph/exp/processed_gpu` | Override for your machine. |
| `--dry-run` | flag | off | |
| `--strict` | flag | off | Exit code 1 if any file skipped. |

### `scripts/copy_gpu_features_valid_time.py`

Copies merged YAMLs whose **`time`** is a positive finite number (drop unlabeled rows).

| Argument | Required | Description |
|----------|----------|-------------|
| `--src` | yes | Source directory. |
| `--dst` | yes | Destination directory. |
| `--dry-run` | no | |

## Next step

Labeled merged directory → `autoconfig train-merged` ([training.md](training.md)).
