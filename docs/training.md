# Training & evaluation (merged 53-D YAML)

Train a regressor on a directory of **merged** feature YAMLs and evaluate on a hold-out or separate test set. Backend is selected with **`--model-kind`** (**`bayesian`**, **`mlp`** / **`nn`**, **`rl`**). Artifacts: `<output>/<model_basename>.pkl` and **`<model_basename>_meta.yaml`** (required for `eval-merged` and `recommend-conf`).

## Labels & layout

- **53-D** `feature_names` / `feature_vector`: `price`(0), `time`(1), `cost`(2), then `static_*`, `sym_*`, graph/partition fields, `conf_*`.
- **Training input X**: indices **3** … end (50 features; “dimensions 4–53” in 1-based wording).
- **Training target Y**: **`-y` / `--y-axis`**: **0** = price, **1** = time (common default from wrappers), **2** = cost.

## CLI: `autoconfig train-merged`

```bash
autoconfig train-merged -h
```

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `-y`, `--y-axis` | `-y` | no | `1` | Must be **`0`**, **`1`**, or **`2`**. |
| `--data-dir` | `-d` | yes | — | Directory of merged feature YAMLs. |
| `--output` | `-o` | no | `out/models` | Directory for `.pkl` and `_meta.yaml`. |
| `--test-split` | — | no | `0.2` | Hold-out fraction in **[0,1]**; **`0`** = train on all files (no internal test split). |
| `--seed` | — | no | `42` | Shuffle seed. |
| `--pattern` | — | no | `*.yaml` | Glob under `data-dir`. |
| `--model-basename` | — | no | `bayesian_cost_merged` | Output stem (**many shell wrappers require an explicit value** — see below). |
| `--n-iter` | — | no | `300` | **`bayesian`**: max variational iterations (also passed as `n_iter` unless overridden in `--model-options`). **`rl`**: used as **`n_epochs`** when `n_epochs` is not set in `--model-options`. **`mlp`** / **`nn`**: ignored by the top-level flag — set e.g. **`max_iter`** in `--model-options`. |
| `--model-kind` | — | no | `bayesian` | **Case-insensitive**: **`bayesian`**, **`mlp`**, **`nn`** (alias of mlp), **`rl`** (Gaussian policy + REINFORCE; see below). |
| `--model-options` | — | no | `None` | **JSON object string** for the backend constructor. **MLP/NN** e.g. `'{"hidden_layer_sizes":[128,64],"max_iter":400,"early_stopping":false}'`; `'{"scale_xy":false}'` turns off `StandardScaler` on inputs (and on `y` when scaling is used). **RL** e.g. `'{"n_epochs":500,"batch_size":64,"learning_rate":0.1,"policy_std":0.35,"scale_xy":true}'`. |
| `--exclude-static` | — | no | off (flag) | Drop `static_*` from X. |
| `--exclude-symbolic` | — | no | off | Drop `sym_*` from X. |
| `--full-graph-with-no-pf` | — | no | off | With static/symbolic excluded, keep full `graph_*` / `partition_*` (legacy “no_pf”); mutually exclusive with |V|/|E|-only ablations — see CLI help. |
| `--graph-ve-only` | — | no | off | **no_graph** ablation: only `graph_num_vertices`, `graph_num_edges`. |
| `--graph-e-only` | — | no | off | Graph block only `graph_num_edges`; often combined with `--exclude-static --exclude-symbolic --graph-ve-only` for **no_all**. |
| `--batch-size` | — | no | `None` | Float: override each row’s **`conf_batch_size`** (I/O batch in config, not SGD mini-batch); recorded in meta. |

### Where `--model-basename` and `--model-kind` are wired

Those flags belong only to the **`train-merged`** subcommand (they are **not** global `autoconfig` options):

| Layer | Role |
|--------|------|
| **CLI** | `autoconfig train-merged --model-basename <stem> --model-kind rl …` (`autoconfig train-merged -h` lists both). Definitions: **`autoconfig/cli.py`** on the `train_merged_parser` (**`--model-basename`** default stem, **`--model-kind`** backend name). |
| **Handler** | **`cmd_train_merged`** reads `args.model_basename`, `args.model_kind`, parses **`--model-options`** JSON, then calls the training pipeline. |
| **Library API** | **`autoconfig.merged.pipeline.train_bayesian_cost_from_merged_yamls`** kwargs **`model_basename`** and **`model_kind`** (`model_kind` selects the regressor via **`autoconfig.models.merged_registry.create_model`**). |
| **`_meta.yaml`** | Writes **`model_kind`** into `{model_basename}_meta.yaml` next to `{model_basename}.pkl`; **`eval-merged`** / **`recommend-conf`** read that sidecar (those commands have **no** **`--model-kind`**). |

**Shell wrappers** (`scripts/run_train_merged*.sh`) forward **`"$@"`** to `train-merged`, so pass the same **`--model-basename`** / **`--model-kind`** flags there—e.g. **`./scripts/run_train_merged.sh --model-basename rl_cost_run1 --data-dir … --output … --model-kind rl`** (most wrappers besides `run_train_merged.sh` / `run_train_merged_price.sh` **require** an explicit **`--model-basename`** via **`scripts/_require_model_basename.inc.sh`**).

Example (direct CLI):

```bash
autoconfig train-merged -d out/train -o out/models \
  --model-basename rl_cost_merged \
  --model-kind rl \
  --n-iter 400
```

## Regression backends (`--model-kind`)

All kinds share the same merged **(X, y)** training API; checkpoints are **`.pkl`** (Bayesian: `pickle`; MLP/NN and RL: **`joblib`** with a `model_kind` field). **`eval-merged`** and **`recommend-conf`** load via `_meta.yaml` + the registry.

### `bayesian` (default)

Variational Bayesian Ridge–style regression (implementation: `BayesianCostModel`). Use **`--n-iter`** or pass **`n_iter`** in **`--model-options`**.

### `mlp` / `nn`

`sklearn` **`MLPRegressor`**, optionally wrapped with **`StandardScaler`** on **X** and target (`scale_xy=true` by default). Hyperparameters via **`--model-options`** (**`hidden_layer_sizes`**, **`max_iter`**, **`learning_rate_init`**, …).

### `rl`

`RLGaussianPolicyRegressor`: each training row is treated as a **one-step MDP** — normalized features as state, a **Gaussian** action in normalized **y**-space, reward **`-(a - y_scaled)²`**, policy gradient (**REINFORCE**) with a moving-average baseline. **Inference uses the deterministic policy mean** (denormalized to the original label scale), so the CLI pipeline matches Bayesian/MLP. **`return_std`** in code reports **policy sampling scale**, not Bayesian epistemic uncertainty.

Typical **`--model-options`** keys: **`n_epochs`**, **`batch_size`**, **`learning_rate`**, **`policy_std`**, **`random_state`**, **`baseline_momentum`**, **`grad_clip_norm`**, **`scale_xy`**. If **`n_epochs`** is omitted, the top-level **`--n-iter`** value is reused as **`n_epochs`**.

Custom environments (online config search, discrete actions over configs, …) should implement **`MergedTabularRegressor`** and **`register_model_kind`** in `autoconfig.models.merged_registry` rather than relying on this tabular surrogate.

## CLI: `autoconfig eval-merged`

```bash
autoconfig eval-merged -h
```

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--model` | `-m` | yes | — | Trained `.pkl`. |
| `-y`, `--y-axis` | `-y` | no | `None` | **If omitted**: read `y_axis` or legacy `target` from sibling `*_meta.yaml`; else **1** (time). If set: only **`0`/`1`/`2`**. |
| `--data-dir` | `-d` | yes | — | Test merged YAML directory. |
| `--pattern` | — | no | `*.yaml` | Glob. |
| `--output` | `-o` | no | `None` | If set, YAML with metrics + `per_file`. |
| `--show-per-file` | — | no | off | Print `y_true`, `y_pred`, `abs_error` per file. |
| `--no-check-feature-names` | — | no | off | Skip name match vs meta (length must still match). |
| `--batch-size` | — | no | `None` | Override `conf_batch_size` before predict (match training if you used an override). |

## Shell wrappers

Run from repo root; prefer **`.venv/bin/autoconfig`**, else **`python -m autoconfig.cli`**.

### Training wrappers (flags below are **prepended** before your args; a later **`-y`** can override the default **`--y-axis`** from the wrapper)

| Script | Injected `train-merged` flags |
|--------|------------------------------|
| `scripts/run_train_merged.sh` | `--y-axis 1` |
| `scripts/run_train_merged_price.sh` | `--y-axis 0` |
| `scripts/run_train_merged_no_graph.sh` | `--y-axis 1 --graph-ve-only` |
| `scripts/run_train_merged_no_graph_sgf.sh` | `--y-axis 1 --graph-ve-only --exclude-symbolic` |
| `scripts/run_train_merged_no_pf.sh` | `--y-axis 1 --exclude-static --exclude-symbolic` |
| `scripts/run_train_merged_no_spf.sh` | `--y-axis 1 --exclude-static` |
| `scripts/run_train_merged_no_sgf.sh` | `--y-axis 1 --exclude-symbolic` |
| `scripts/run_train_merged_no_all.sh` | `--y-axis 1 --exclude-static --exclude-symbolic --graph-ve-only --graph-e-only` |

Except **`run_train_merged.sh`** and **`run_train_merged_price.sh`**, these **`source scripts/_require_model_basename.inc.sh`**: you **must** pass a non-empty **`--model-basename`** (or `--model-basename=value`) or the script exits with a usage hint.

`_require_model_basename.inc.sh` takes no arguments; it is only sourced.

### Evaluation wrappers

| Script | Behavior |
|--------|----------|
| `scripts/run_eval_merged.sh` | **`$#==0`**: default `-m out/models/bayesian_cost_merged.pkl -d out/test`; else **`eval-merged "$@"`**. |
| `scripts/run_eval_merged_price.sh` | **`$#==0`**: default `-m out/models/price/bayesian_price_merged.pkl -d out/test/price`; else **`eval-merged --y-axis 0 "$@"`** (a later `-y` overrides). |

## Other scripts

### `scripts/train_pipeline.sh`

**Legacy**: invokes `python -m autoconfig train` (there is **no** `train` subcommand today). For the merged pipeline use **`train-merged`** or **`run_train_merged.sh`**.

### `scripts/run_exp_conf_gpu_benchmarks.sh`

Runs GridGraph (wcc/bfs/pr) and MatrixGraph subiso against `exp/conf/gpu/conf_NN.yaml`; **no positional args** — only **environment variables**:

| Variable | Default | Description |
|----------|---------|-------------|
| `AC_CONF` | `$ROOT/exp/conf/gpu` | AutoConfig config directory. |
| `GRIDGRAPH_ROOT` | `/home/zhuxk/project/graph/GridGraph` | GridGraph source root. |
| `MG_ROOT` | `/home/zhuxk/project/graph/MatrixGraph` | MatrixGraph root. |
| `GRIDGRAPH_WORKSPACE` | `/ssd_data/zhuxk/workspace/GridGraph_workspace` | Graph datasets parent. |
| `GRAPHS` | `friendster livejournal patents web-sk` | Dataset names. |
| `MIN_CONF`, `MAX_CONF` | `1`, `50` | Inclusive `conf_NN` range. |
| `SKIP_GRIDGRAPH` | empty | If non-empty, skip wcc/bfs/pr. |
| `SKIP_SUBISO` | empty | If non-empty, skip subiso. |
| `MG_PATTERN`, `MG_DATA` | empty | Passed as `-p` / `-g` to `MatrixGraph/scripts/run_autoconfig_subiso_exp.py`. |
| `OUT_ROOT` | `$ROOT/exp/gpu` | Output root. |
| `RESTORE_PARALLELISM` | `44` | Restore `parallelism` in GridGraph `graph.hpp` after runs. |
| `PYTHON` | auto `.venv/bin/python` or `python3` | Interpreter. |

## Next step

After training → `.pkl` + `_meta.yaml` → [conf-recommend.md](conf-recommend.md).
