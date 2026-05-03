# Config recommendation (`recommend-conf`)

Given a **trained** merged model, **query** and **graph** YAML, and **candidate configs** (single YAML with multiple rows or a directory of one YAML per config), rank configs by the predicted target (from training `y_axis` / `_meta.yaml`). Optionally export the top **`k`** as standalone YAML files.

## CLI: `autoconfig recommend-conf`

```bash
autoconfig recommend-conf -h
```

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--model` | `-m` | yes | — | `train-merged` `.pkl`; sibling **`<stem>_meta.yaml`** required. |
| `--query` | `-q` | yes | — | Query-feature YAML. |
| `--graph` | `-g` | yes | — | Graph-feature YAML. |
| `--config` | `-c` | yes | — | **One** merge-format YAML (multiple `configurations`) **or** a **directory** of `*.yaml` (one config per file; no subdirectories). |
| `--k` | `-k` | no | `5` | Return top **N** configs (integer). |
| `--maximize` | — | no | off | Default **minimize** predicted value; if set, **maximize**. |
| `--batch-size` | — | no | `None` | Override `conf_batch_size` on merged rows (match training override if used). |
| `--output` | `-o` | no | `None` | **`DIR_OR_FILE`**: **directory** → `recommend_rank01.yaml`, …; **`.yaml`/`.yml` file** → single summary (`to_dict`). |
| `--export-prefix` | — | no | `recommend` | Filename prefix under `-o` dir. |
| `--full-report` | — | no | off | With `-o` a directory, also write `{prefix}_full_report.yaml`. |
| `--refine` | — | no | off | Local search after top-k (random perturb cpu/memory/GPU; see `autoconfig.conf_recommend.refine_search`). |
| `--refine-max-iterations` | — | no | `10000` | Max refine rounds. |
| `--refine-seed` | — | no | `None` | Optional RNG seed. |

## Shell wrappers

| Script | Behavior |
|--------|----------|
| `scripts/run_recommend_conf.sh` | Prints usage and exits **1** with no args; else **`autoconfig recommend-conf "$@"`**. |
| `scripts/run_recommend_conf_dir.sh` | Same forwarding; with **`$#==0`**, reads env vars below; if **`RECOMMEND_EXPORT_DIR`** is set and `-o` is missing, appends **`-o`**. **`GRAPH_YAML`** must be set for the no-arg mode. |
| `scripts/run_conf_recommend.sh` | `exec run_recommend_conf.sh`. |
| `scripts/run_conf_recommend_dir.sh` | `exec run_recommend_conf_dir.sh`. |

### `run_recommend_conf_dir.sh` environment variables (**only when `$# == 0`**)

| Variable | Default | Maps to |
|----------|---------|---------|
| `MERGED_MODEL` | `out/models/bayesian_cost_merged.pkl` | `-m` |
| `QUERY_YAML` | `out/query_features/kernel_bfs.yaml` | `-q` |
| `GRAPH_YAML` | *empty* | Must be set or use explicit CLI args |
| `CONF_DIR` | `data/conf/gpu` | `-c` as `${CONF_DIR}/` |
| `K` | `5` | `-k` |
| `RECOMMEND_EXPORT_DIR` | empty | If set, append `-o` |

## Smoke tests (no large hand-written YAML)

| Script | Runs | CLI args |
|--------|------|----------|
| `scripts/recommend_conf_smoke.sh` | `recommend_conf_smoke.py` | None (internal tempfile + minimal YAML). |
| `scripts/recommend_conf_smoke_dir.sh` | `recommend_conf_smoke_dir.py` | None; tests **`-c` as directory** so each rank has `source_path`. |

The Python entry points do **not** parse command-line arguments (fixed `main()` only).

## Relation to training

The **`-m`** model must match merged feature dimensionality for the same query/graph/config layout; **`target` / `y_axis`** from `*_meta.yaml` defines what “better” means (time / price / cost).
