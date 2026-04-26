# Merged features: training and evaluation

This document describes how to **train** a Bayesian cost regressor and **evaluate** it on a held-out folder, using the **merged feature YAML** format produced by `autoconfig merge` (or equivalent tooling). Paths below are **examples**; all CLI arguments accept your own locations.

**Related reading:** [FEATURE_PIPELINE.md](FEATURE_PIPELINE.md) (merge step), [feature_extraction.md](feature_extraction.md) (vector layout), [CLI_GUIDE.md](CLI_GUIDE.md) (all `autoconfig` commands).

### Label `y` in the CLI (`-y` / `--y-axis`)

The command line only needs an integer; the **name** of that column in the YAML is fixed by convention:

| `y` | Column in `feature_names` (slot 1–3) |
|-----|--------------------------------------|
| **0** | `price` |
| **1** | `time` (default for `train-merged` if you omit `-y`) |
| **2** | `cost` |

Short form: **`-y N`** (same as **`--y-axis N`**). The merged file still **stores** the scalars as `price` / `time` / `cost`; only the **CLI** uses `0/1/2`. Metadata `*_meta.yaml` records both `y_axis` and a human-readable `target` name.

---

## Data layout

Each training or test item is one YAML file with:

- `feature_names`: list of length **53**, fixed order from :class:`~autoconfig.utils.feature_merger.FeatureMerger`: **positions 1–3 (1-based)** are ``price``, ``time``, ``cost``; **positions 4–53** are static (8), symbolic (12), graph+partition (17), config (13, including ``conf_price``). At merge, ``price`` and ``conf_price`` come from the config catalog; ``time`` and ``cost`` are **0** until you fill them (e.g. set ``time`` from GridGraph ``STATS``).
- `feature_vector`: same length as `feature_names`. **Training and evaluation use a fixed layout:**
  - **X (inputs)** = **features 4–53 only** (0-based indices ``3..52``) — program, graph, and config features. The other two of ``price`` / ``time`` / ``cost`` are **not** part of `X` when you predict a different label (no leakage of the other objectives into inputs).
  - **Y (label)** = exactly one of the first three columns, chosen with **``-y``** / **``--y-axis``** ``0|1|2`` (see table above).

All files in a directory must use the **same** `feature_names` list (in the same order). Typical layout:

- **Train directory:** e.g. `out/train/`, a glob of `*.yaml` (or a narrower `pattern` such as `*_gridgraph_wcc.yaml`).
- **Test directory:** the same format, e.g. `out/test/`, for reporting MAE, RMSE, MAPE, and R² on unseen examples.

Filling the `cost` (and `time`) slots for training often comes from measured runs; for evaluation, the **label column** you trained on must be present in each YAML (e.g. if you train on `time`, `time` must be filled in test files).

---

## Scripts: from query/graph/config YAMLs to training

| Step | Script / command | Role |
|------|------------------|------|
| 1. **Grid of merged feature files** (one file per `conf×graph×query`) | `python scripts/merge_abc_features.py` | **Required args:** `--query-dir`, `--graph-dir`, `--config-dir`, `--out-dir`. Produces **53‑D** YAMLs; `price` / `conf_price` from catalog, `time`/`cost` = 0 until filled. **Optional** `--query-stems` (see `--help`). |
| 2. **Set `time` from measured runs** | `python scripts/fill_merged_train_costs_from_stats.py` | Writes **`time`** from `GridGraph/.../STATS.txt` **`real`** column. **Default** `--gridgraph-root` = sibling `GridGraph/`. Optional **核/内存** check vs `config_scalars`. In‑place: set `--output-dir` = `--features-dir`. |
| 3. **Train** | `scripts/run_train_merged.sh` or `autoconfig train-merged` | Default **`-y 1`** (time); `X` = features 4–53. |
| 4. **Test / report** | `autoconfig eval-merged` or `scripts/run_eval_merged.sh` | Same layout; Y from **`-y`** or ``*_meta.yaml`` (`y_axis` / legacy `target` string). |

**`autoconfig merge`** (single triple: one query, one graph, one config file with possibly many rows) is an alternative to step 1 when you do not use the A×B×C batch helper.

`merge_abc_features.py` is documented in its own file header; `fill_merged_train_costs_from_stats.py` is documented in its module docstring.

---

## Environment

The `autoconfig` entry point is available after a local install. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

If `autoconfig: command not found`, either activate the venv or run:

```bash
.venv/bin/python -m autoconfig.cli <subcommand> ...
```

Helper scripts under `scripts/` (see below) call the venv the same way.

---

**X** is always the **50** features from index 3 onward in `feature_names` (after the fixed ``price``, ``time``, ``cost`` prefix), before optional ablation flags.

Helpers for **catalog price** as Y (`y=0`): ``./scripts/run_train_merged_price.sh`` / ``./scripts/run_eval_merged_price.sh`` (they pass ``--y-axis 0``).

---

## Training: `train-merged`

Fits a **Bayesian** cost model (`BayesianCostModel` in the codebase: log target, normalized inputs, variational fit). It writes:

- `bayesian_cost_merged.pkl` (or a name you choose) — the trained model
- `bayesian_cost_merged_meta.yaml` — training metadata, including `feature_names_x`, and train/test metrics when a split is used

**Basic usage:**

```bash
autoconfig train-merged --data-dir out/train --output out/models
```

**Useful options:**

| Option | Description |
|--------|-------------|
| `--data-dir`, `-d` | Directory of merged feature YAMLs (required) |
| `--output`, `-o` | Where to write `.pkl` and `*_meta.yaml` (default `out/models`) |
| `--test-split` | Fraction of files reserved for a random test split, for printed metrics (default `0.2`). Use `0` to train on **all** files (no internal test) |
| `--seed` | RNG seed for the split (default `42`) |
| `--pattern` | Glob under `data-dir` (default `*.yaml`), e.g. `*_gridgraph_wcc.yaml` to train on one kernel family only |
| `--model-basename` | Basename for output files (default `bayesian_cost_merged`) |
| `-y`, `--y-axis` | `0` / `1` / `2` = label column (default **`1`** = time; see table at top) |
| `--n-iter` | Max **variational** steps in the Bayesian fit (default `300`; larger may refine longer before `tol` stops) |
| `--exclude-static` | Drop static program features (names starting with `static_`), i.e. SPF |
| `--exclude-symbolic` | Drop graph-parameterized symbolic features (names starting with `sym_`), i.e. SGF |

**Ablation scripts** (convenience wrappers; they set a distinct `--model-basename` and pass `--data-dir out/train --output out/models --test-split 0` when you run them with no arguments):

| Script | Effect | Default output basename |
|--------|--------|-------------------------|
| `scripts/run_train_merged_no_spf.sh` | `--exclude-static` | `bayesian_cost_merged_no_spf` |
| `scripts/run_train_merged_no_sgf.sh` | `--exclude-symbolic` | `bayesian_cost_merged_no_sgf` |
| `scripts/run_train_merged_no_pf.sh` | both flags (only **graph** + **config** columns) | `bayesian_cost_merged_no_pf` |

**Shell helper** (default: `--data-dir out/train` and `--output out/models`; all extra flags are forwarded to `autoconfig train-merged`):

```bash
chmod +x scripts/run_train_merged.sh
./scripts/run_train_merged.sh
# or
./scripts/run_train_merged.sh --data-dir out/train --output out/models --test-split 0
./scripts/run_train_merged.sh --n-iter 500
```

---

## Evaluation (test): `eval-merged`

Loads a trained `.pkl` and a **test directory** of merged YAMLs, predicts the trained label (selected by the same `y` as in training), and compares to that column in each file. Prints aggregate metrics; optionally writes a YAML report and lists per-file errors.

**Basic usage:**

```bash
autoconfig eval-merged \
  --model out/models/bayesian_cost_merged.pkl \
  --data-dir out/test
```

**Options:**

| Option | Description |
|--------|-------------|
| `--model`, `-m` | Path to the `.pkl` from `train-merged` (required) |
| `--data-dir`, `-d` | Test folder of merged YAMLs (required) |
| `-y`, `--y-axis` | Optional override (same `0|1|2` as training). If omitted, uses `y_axis` or legacy `target` in `*_meta.yaml`; if still unknown, default **`1`** (time) |
| `--pattern` | Same as training (default `*.yaml`) |
| `--output`, `-o` | Optional path to write the full result dict (metrics + `per_file` list) as YAML |
| `--show-per-file` | Also print one line per file: `y_true`, `y_pred`, `abs_error` |
| `--no-check-feature-names` | Do not require `feature_names_x` to match the model meta (dimension must still match). Use only if you know the layout is equivalent |

A `*_meta.yaml` next to the model is used, when present, to apply the same **feature exclusions** as training (`exclude_static` / `exclude_symbolic` are read from the meta), then to verify that the filtered `feature_names_x` and dimension match the model. You do not need to pass extra flags to `eval-merged` for ablation models: use the correct `.pkl` and its sidecar metadata.

**Shell helper:**

```bash
chmod +x scripts/run_eval_merged.sh
./scripts/run_eval_merged.sh -m out/models/bayesian_cost_merged.pkl -d out/test
./scripts/run_eval_merged.sh -m out/models/bayesian_cost_merged.pkl -d out/test -o out/eval_report.yaml
```

---

## Metrics reported

For evaluation, the **same** definitions are used as in training-side reporting:

- **MAE** — mean absolute error
- **RMSE** — root mean square error
- **MAPE (%)** — mean absolute percentage error, with a small floor on `y_true` to avoid division by zero
- **R²** — coefficient of determination relative to the mean of `y` on the evaluated set

---

## Python API

For scripts or notebooks:

```python
from pathlib import Path
from autoconfig.offline.yaml_feature_trainer import (
    load_merged_feature_dir,
    train_bayesian_cost_from_merged_yamls,
    evaluate_bayesian_cost_on_merged_dir,
)

# Train (example paths)
train_bayesian_cost_from_merged_yamls(
    Path("out/train"),
    Path("out/models"),
    test_split=0.2,
    pattern="*.yaml",
    model_basename="bayesian_cost_merged",
    y_axis=1,  # 0=price, 1=time, 2=cost
    n_iter=300,
    verbose=True,
)

# Evaluate
result = evaluate_bayesian_cost_on_merged_dir(
    "out/models/bayesian_cost_merged.pkl",
    "out/test",
    pattern="*.yaml",
    y_axis=1,  # must match the model / training; or omit and use defaults
    check_feature_names=True,
)
print(result["metrics"])
# result["per_file"] — list of {file, y_true, y_pred, abs_error}
```

---

## Differences from `autoconfig train`

The default **`autoconfig train`** subcommand uses the **legacy offline pipeline** (synthetic `DataGenerator` samples and a different feature shape). The **`train-merged` / `eval-merged`** flow is the one to use for **real merged 53-dimensional** vectors (``price``, ``time``, ``cost``, then 50 inputs) from the YAML merge pipeline.

---

## See also

- [CLI_GUIDE.md](CLI_GUIDE.md) — install and all commands
- [docs/README.md](README.md) — documentation index
